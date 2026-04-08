#!/usr/bin/env python3
"""
DataQA Inference Script — Two-Phase Agent
------------------------------------------
LLM agent that plays the DataQA environment in two phases:
  Phase 1: Identify all data quality issues
  Phase 2: Propose fixes for identified issues

Uses the OpenAI client to interact with any OpenAI-compatible LLM API.

Required environment variables:
    API_BASE_URL  - LLM API endpoint (e.g., https://router.huggingface.co/v1)
    MODEL_NAME    - Model identifier (e.g., Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      - HuggingFace token / API key

STDOUT FORMAT (mandatory for evaluation):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK = "dataqa_env"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = 3


# ---------------------------------------------------------------------------
# Logging helpers (structured stdout — exact format required by evaluation)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP client
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

    def step(self, issues: list[str], fixes: list[str], task_id: str = "easy") -> dict:
        r = self.session.post(
            f"{self.base_url}/step",
            json={"action": {"issues": issues, "fixes": fixes, "task_id": task_id}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

IDENTIFY_SYSTEM_PROMPT = """You are a data quality analyst. Your job is to inspect datasets and identify data quality issues.

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
- DO NOT use the employee_id, order_id, or experiment_id as the row number
- Column names must match exactly (use the CSV header names, lowercase)
- Check EVERY row and EVERY column systematically
- Consider cross-column consistency (e.g., total = quantity * price)
- Look for subtle issues like whitespace-only values, near-duplicates
- Report ALL issues you find, even if uncertain

Respond with ONLY the list of issues, one per line. No other text.
Example: row:3,col:salary,issue:missing_value"""


FIX_SYSTEM_PROMPT = """You are a data repair specialist. You have already identified data quality issues in a dataset. Now you must propose the correct values to fix each issue.

For each issue you identified, propose a fix in EXACTLY this format:
row:<row_number>,col:<column_name>,fix:<corrected_value>

Guidelines for proposing fixes:
- For missing_value: infer the correct value from context, schema, and other rows
- For wrong_type: convert to the correct type (e.g., "seventy-five thousand" → "75000")
- For out_of_range: propose a value within the valid range that makes sense in context
- For format_violation: correct the format (e.g., "26/01/2024" → "2024-01-26")
- For inconsistent_value: compute the correct value from related fields
- For duplicate_row: propose a corrected unique key or indicate removal
- For statistical_outlier: propose a reasonable value given the model/context

Use the schema, validation rules, and surrounding data to determine the correct fix.
Respond with ONLY the list of fixes, one per line. No other text.
Example: row:3,col:salary,fix:75000"""


def build_user_prompt(observation: dict, include_fixes: bool = False) -> str:
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

    if include_fixes:
        parts.append(
            "Now propose fixes for ALL issues. "
            "Use format: row:<N>,col:<name>,fix:<corrected_value>"
        )

    return "\n\n".join(parts)


def parse_llm_response(response: str) -> list[str]:
    """Extract issue lines from LLM response."""
    issues = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[\d]+[.\)]\s*", "", line)
        line = re.sub(r"^\s*[-*]\s*", "", line)
        line = line.strip()
        if "row" in line.lower() and "col" in line.lower():
            match = re.search(
                r"row\s*[:=]\s*(\d+)\s*[,;\s]+col(?:umn)?\s*[:=]\s*([\w_]+)\s*[,;\s]+issue\s*[:=]\s*([\w_]+)",
                line,
                re.IGNORECASE,
            )
            if match:
                normalized = f"row:{match.group(1)},col:{match.group(2).lower()},issue:{match.group(3).lower()}"
                issues.append(normalized)
    return issues


def parse_fix_response(response: str) -> list[str]:
    """Extract fix lines from LLM response."""
    fixes = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[\d]+[.\)]\s*", "", line)
        line = re.sub(r"^\s*[-*]\s*", "", line)
        line = line.strip()
        if "row" in line.lower() and "fix" in line.lower():
            match = re.search(
                r"row\s*[:=]\s*(\d+)\s*[,;\s]+col(?:umn)?\s*[:=]\s*([\w_]+)\s*[,;\s]+fix\s*[:=]\s*(.+?)$",
                line,
                re.IGNORECASE,
            )
            if match:
                normalized = f"row:{match.group(1)},col:{match.group(2).lower()},fix:{match.group(3).strip()}"
                fixes.append(normalized)
    return fixes


def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call the LLM with retry on rate limit."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 10 * (attempt + 1)
                print(f"[DEBUG] Rate limited, waiting {wait}s...", file=sys.stderr, flush=True)
                time.sleep(wait)
            else:
                print(f"[DEBUG] LLM call failed: {e}", file=sys.stderr, flush=True)
                return ""
    return ""


def run_task(client: OpenAI, env: EnvHTTPClient, task_id: str) -> float:
    """
    Run a single task with two-phase strategy:
      Step 1: Identify issues only
      Step 2: Identify + Fix (using feedback from step 1)
      Step 3: Refined identify + fix (if needed)
    """
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    best_score = 0.0
    success = False

    try:
        reset_response = env.reset(task_id=task_id)
        observation = reset_response.get("observation", reset_response)

        last_issues: list[str] = []
        last_llm_output = ""

        for step_num in range(1, MAX_STEPS_PER_TASK + 1):
            error_msg = None

            # ── Phase 1: Identify issues ──
            user_prompt = build_user_prompt(observation)
            identify_output = call_llm(client, IDENTIFY_SYSTEM_PROMPT, user_prompt)
            issues = parse_llm_response(identify_output)

            if not issues and not error_msg:
                error_msg = "no issues parsed from LLM response"

            # ── Phase 2: Propose fixes (from step 2 onward, or always if we have issues) ──
            fixes: list[str] = []
            if issues and step_num >= 2:
                # Build a fix prompt that includes the identified issues
                fix_prompt = build_user_prompt(observation, include_fixes=True)
                fix_prompt += f"\n\nISSUES FOUND:\n" + "\n".join(issues)
                fix_output = call_llm(client, FIX_SYSTEM_PROMPT, fix_prompt)
                fixes = parse_fix_response(fix_output)

            # ── Submit to environment ──
            action_str = ";".join(issues[:5]) if issues else "none"
            if fixes:
                action_str += "|fixes:" + ";".join(fixes[:3])

            step_response = env.step(issues, fixes, task_id=task_id)
            observation = step_response.get("observation", step_response)

            reward = float(step_response.get("reward", 0.0) or 0.0)
            done = bool(step_response.get("done", False))
            best_score = max(best_score, reward)
            rewards.append(reward)
            steps_taken = step_num

            log_step(
                step=step_num,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

            last_issues = issues
            last_llm_output = identify_output

        success = best_score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, score=best_score, rewards=rewards)

    return best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[DEBUG] DataQA Inference starting", file=sys.stderr, flush=True)
    print(f"[DEBUG] ENV_URL={ENV_URL}", file=sys.stderr, flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", file=sys.stderr, flush=True)

    env = EnvHTTPClient(ENV_URL)
    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY or "no-key",
    )

    if not env.health():
        print("[DEBUG] Environment is not healthy. Exiting.", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[DEBUG] Environment is healthy", file=sys.stderr, flush=True)

    scores = {}
    for task_id in TASKS:
        try:
            score = run_task(llm_client, env, task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"[DEBUG] Task {task_id} failed: {e}", file=sys.stderr, flush=True)
            scores[task_id] = 0.0

    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n[DEBUG] FINAL RESULTS: {scores} avg={avg_score:.3f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
