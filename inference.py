#!/usr/bin/env python3
"""
DataQA Inference Script
-----------------------
LLM agent that plays the DataQA environment.
Uses the OpenAI client to interact with any OpenAI-compatible LLM API.

Required environment variables:
    API_BASE_URL  - LLM API endpoint (e.g., https://api.groq.com/openai/v1)
    MODEL_NAME    - Model identifier (e.g., llama-3.3-70b-versatile)
    HF_TOKEN      - HuggingFace token (for HF Spaces access)

Structured logging format: [START], [STEP], [END] tags for evaluation.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = 3

# ---------------------------------------------------------------------------
# Logging helpers (structured stdout for evaluation)
# ---------------------------------------------------------------------------

def log_start(task_id: str, metadata: Optional[dict] = None):
    entry = {"event": "START", "task_id": task_id, "timestamp": time.time()}
    if metadata:
        entry["metadata"] = metadata
    print(f"[START] {json.dumps(entry)}", flush=True)


def log_step(task_id: str, step: int, reward: float, details: Optional[dict] = None):
    entry = {
        "event": "STEP",
        "task_id": task_id,
        "step": step,
        "reward": reward,
        "timestamp": time.time(),
    }
    if details:
        entry["details"] = details
    print(f"[STEP] {json.dumps(entry)}", flush=True)


def log_end(task_id: str, final_score: float, metadata: Optional[dict] = None):
    entry = {
        "event": "END",
        "task_id": task_id,
        "final_score": final_score,
        "timestamp": time.time(),
    }
    if metadata:
        entry["metadata"] = metadata
    print(f"[END] {json.dumps(entry)}", flush=True)


# ---------------------------------------------------------------------------
# Environment HTTP client (simple, no WebSocket needed for inference)
# ---------------------------------------------------------------------------

class EnvHTTPClient:
    """Minimal HTTP client for the DataQA environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task_id: str = "easy") -> dict:
        r = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(self, issues: list[str], task_id: str = "easy") -> dict:
        r = self.session.post(
            f"{self.base_url}/step",
            json={"action": {"issues": issues, "task_id": task_id}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = self.session.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a data quality analyst. Your job is to inspect datasets and identify data quality issues.

You will be given:
1. A dataset in CSV format
2. A schema describing expected column types and constraints
3. Validation rules that the data should satisfy

You must identify ALL data quality issues and report each one in EXACTLY this format:
row:<row_number>,col:<column_name>,issue:<issue_type>

Supported issue types:
- missing_value (null, empty, or whitespace-only)
- wrong_type (value doesn't match expected type)
- duplicate_row (exact duplicate or duplicate key)
- out_of_range (value outside valid range)
- format_violation (wrong format, invalid enum value)
- inconsistent_value (computed field doesn't match, logical inconsistency)
- statistical_outlier (value is unreasonable given context)
- referential_integrity (foreign key violation)

CRITICAL INSTRUCTIONS FOR ROW NUMBERING:
- Row numbers refer to the ROW POSITION in the CSV data, NOT the value of any ID column
- Row 1 = the FIRST data row after the header
- Row 2 = the SECOND data row after the header
- For example, if the CSV has header on line 1 and data starting on line 2, the data on line 2 is row 1, line 3 is row 2, etc.
- DO NOT use the employee_id, order_id, or experiment_id as the row number
- Column names must match exactly (use the CSV header names, lowercase)
- Check EVERY row and EVERY column systematically
- Consider cross-column consistency (e.g., total = quantity * price)
- Look for subtle issues like whitespace-only values, near-duplicates
- Report ALL issues you find, even if uncertain

Respond with ONLY the list of issues, one per line. No other text.
Example: row:3,col:salary,issue:missing_value"""


def build_user_prompt(observation: dict) -> str:
    obs = observation if isinstance(observation, dict) else observation
    parts = []

    if obs.get("task_description"):
        parts.append(f"TASK: {obs['task_description']}")

    parts.append(f"SCHEMA:\n{obs.get('schema_description', '')}")
    parts.append(f"VALIDATION RULES:\n{obs.get('validation_rules', '')}")
    parts.append(f"DATASET:\n{obs.get('dataset_csv', '')}")

    hint = obs.get("num_issues_hint", 0)
    if hint:
        parts.append(f"HINT: There are exactly {hint} issues to find.")

    feedback = obs.get("feedback", "")
    if feedback and "reset" not in feedback.lower():
        parts.append(f"FEEDBACK FROM PREVIOUS ATTEMPT:\n{feedback}")

    return "\n\n".join(parts)


def parse_llm_response(response: str) -> list[str]:
    """Extract issue lines from LLM response."""
    issues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering like "1. " or "- " or "* "
        line = re.sub(r"^\s*[\d]+[.\)]\s*", "", line)
        line = re.sub(r"^\s*[-*]\s*", "", line)
        line = line.strip()
        if "row" in line.lower() and "col" in line.lower():
            # Lenient regex: accept : or = as delimiters, case-insensitive
            match = re.search(
                r"row\s*[:=]\s*(\d+)\s*[,;\s]+col(?:umn)?\s*[:=]\s*([\w_]+)\s*[,;\s]+issue\s*[:=]\s*([\w_]+)",
                line,
                re.IGNORECASE,
            )
            if match:
                # Normalize to lowercase canonical format
                normalized = f"row:{match.group(1)},col:{match.group(2).lower()},issue:{match.group(3).lower()}"
                issues.append(normalized)
    return issues


def run_task(client: OpenAI, env: EnvHTTPClient, task_id: str) -> float:
    """Run a single task and return the best score."""
    log_start(task_id)

    # Reset environment for this task
    reset_response = env.reset(task_id=task_id)
    observation = reset_response.get("observation", reset_response)

    best_score = 0.0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step_num in range(1, MAX_STEPS_PER_TASK + 1):
        user_prompt = build_user_prompt(observation)
        messages_for_call = messages + [{"role": "user", "content": user_prompt}]

        # Call LLM with retry on rate limit
        llm_output = ""
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages_for_call,
                    temperature=0.1,
                    max_tokens=2048,
                )
                llm_output = response.choices[0].message.content or ""
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 10 * (attempt + 1)
                    print(f"[WARN] Rate limited, waiting {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"[ERROR] LLM call failed: {e}", file=sys.stderr, flush=True)
                    break

        # Parse issues from LLM response
        issues = parse_llm_response(llm_output)

        if not issues:
            print(f"[WARN] No issues parsed from LLM response for {task_id} step {step_num}", file=sys.stderr, flush=True)

        # Submit to environment
        step_response = env.step(issues, task_id=task_id)
        observation = step_response.get("observation", step_response)

        # reward and done are at the top level of the response, not inside observation
        reward = float(step_response.get("reward", 0.0) or 0.0)
        done = bool(step_response.get("done", False))
        best_score = max(best_score, reward)

        log_step(task_id, step_num, reward, {
            "issues_reported": len(issues),
            "feedback": observation.get("feedback", ""),
        })

        if done:
            break

        # Add context for next attempt
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": llm_output})

    log_end(task_id, best_score)
    return best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[INFO] DataQA Inference starting", flush=True)
    print(f"[INFO] ENV_URL={ENV_URL}", flush=True)
    print(f"[INFO] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL_NAME={MODEL_NAME}", flush=True)

    # Initialize clients
    env = EnvHTTPClient(ENV_URL)
    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=os.environ.get("LLM_API_KEY", HF_TOKEN or "no-key"),
    )

    # Check environment health
    if not env.health():
        print("[ERROR] Environment is not healthy. Exiting.", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[INFO] Environment is healthy", flush=True)

    # Run all tasks
    scores = {}
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"[INFO] Starting task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            score = run_task(llm_client, env, task_id)
            scores[task_id] = score
            print(f"[INFO] Task {task_id} completed with score: {score:.3f}", flush=True)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr, flush=True)
            scores[task_id] = 0.0

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("[INFO] FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for task_id, score in scores.items():
        print(f"[INFO] {task_id}: {score:.3f}", flush=True)

    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"[INFO] Average score: {avg_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
