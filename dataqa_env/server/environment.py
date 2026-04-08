"""
DataQA Environment
------------------
Server-side environment for data quality assurance tasks.

The agent receives corrupted datasets and must identify planted quality issues.
Scoring is based on F1 (precision-recall) of correctly matched issues.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional, Set

from openenv.core.env_server.interfaces import Action, Environment, Observation

from ..models import DataQAAction, DataQAObservation, DataQAState
from .tasks import PlantedIssue, Task, get_task, list_tasks


def parse_issue_key(raw: str) -> Optional[str]:
    """
    Parse an agent-reported issue string into a normalized key.
    Expected format: row:<N>,col:<name>,issue:<type>
    Returns normalized key or None if unparseable.
    """
    raw = raw.strip().lower()
    # Be lenient with formatting
    row_match = re.search(r"row\s*[:=]\s*(\d+)", raw)
    col_match = re.search(r"col\s*[:=]\s*([\w_]+)", raw)
    issue_match = re.search(r"issue\s*[:=]\s*([\w_]+)", raw)

    if row_match and col_match and issue_match:
        return f"row:{row_match.group(1)},col:{col_match.group(1)},issue:{issue_match.group(1)}"
    return None


def compute_f1(reported_keys: Set[str], planted_keys: Set[str]) -> dict:
    """Compute precision, recall, and F1 score."""
    if not reported_keys and not planted_keys:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    if not reported_keys:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(planted_keys)}

    if not planted_keys:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(reported_keys), "fn": 0}

    tp = len(reported_keys & planted_keys)
    fp = len(reported_keys - planted_keys)
    fn = len(planted_keys - reported_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_weighted_reward(
    reported_keys: Set[str],
    planted_issues: list,
) -> dict:
    """
    Compute difficulty-weighted reward for richer per-step signal.

    Each planted issue has a difficulty weight (1.0-3.0). Finding harder issues
    earns more reward. False positives incur a penalty scaled by average difficulty.

    Returns dict with weighted_reward (0.0-1.0), plus per-issue breakdown.
    """
    if not planted_issues and not reported_keys:
        return {"weighted_reward": 1.0, "difficulty_found": 0.0, "difficulty_missed": 0.0}

    planted_by_key = {issue.to_key(): issue for issue in planted_issues}
    planted_keys = set(planted_by_key.keys())

    if not reported_keys:
        total_weight = sum(i.difficulty for i in planted_issues)
        return {"weighted_reward": 0.0, "difficulty_found": 0.0, "difficulty_missed": total_weight}

    if not planted_keys:
        return {"weighted_reward": 0.0, "difficulty_found": 0.0, "difficulty_missed": 0.0}

    # Sum difficulty weights for found vs missed issues
    found_keys = reported_keys & planted_keys
    missed_keys = planted_keys - reported_keys
    false_positive_count = len(reported_keys - planted_keys)

    difficulty_found = sum(planted_by_key[k].difficulty for k in found_keys)
    difficulty_missed = sum(planted_by_key[k].difficulty for k in missed_keys)
    total_weight = sum(i.difficulty for i in planted_issues)

    # Weighted recall: proportion of difficulty captured
    weighted_recall = difficulty_found / total_weight if total_weight > 0 else 0.0

    # Penalty for false positives (scaled by avg difficulty so penalty is proportional)
    avg_difficulty = total_weight / len(planted_issues)
    fp_penalty_weight = false_positive_count * avg_difficulty
    weighted_precision = difficulty_found / (difficulty_found + fp_penalty_weight) if (difficulty_found + fp_penalty_weight) > 0 else 0.0

    # Weighted F1
    if (weighted_precision + weighted_recall) > 0:
        weighted_reward = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
    else:
        weighted_reward = 0.0

    return {
        "weighted_reward": round(weighted_reward, 4),
        "difficulty_found": round(difficulty_found, 2),
        "difficulty_missed": round(difficulty_missed, 2),
    }


class DataQAEnvironment(Environment):
    """
    Data Quality Assurance environment.

    The agent inspects corrupted datasets and reports quality issues.
    Reward is F1 score of correctly identified issues vs planted ground truth.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = DataQAState()
        self._current_task: Optional[Task] = None
        self._planted_keys: Set[str] = set()
        self._best_score: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        task_id = kwargs.get("task_id", "easy")
        task_seed = seed if seed is not None else 42

        self._current_task = get_task(task_id, seed=task_seed)
        self._planted_keys = {issue.to_key() for issue in self._current_task.planted_issues}
        self._best_score = 0.0

        ep_id = episode_id or str(uuid.uuid4())
        self._state = DataQAState(
            episode_id=ep_id,
            step_count=0,
            task_id=task_id,
            current_step=0,
            max_steps=self._current_task.max_steps,
            best_score=0.0,
            total_planted_issues=len(self._current_task.planted_issues),
        )

        return DataQAObservation(
            dataset_csv=self._current_task.corrupted_csv,
            schema_description=self._current_task.schema_description,
            validation_rules=self._current_task.validation_rules,
            task_description=self._current_task.description,
            feedback="Environment reset. Inspect the dataset and report all quality issues.",
            task_id=task_id,
            num_issues_hint=len(self._current_task.planted_issues),
            max_steps=self._current_task.max_steps,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        if not isinstance(action, DataQAAction):
            raise ValueError(f"Expected DataQAAction, got {type(action)}")

        # In stateless HTTP mode, each request creates a fresh env instance.
        # Auto-reset using the task_id from the action so step() works standalone.
        if self._current_task is None:
            self.reset(task_id=action.task_id)

        self._state.step_count += 1
        self._state.current_step += 1

        # Parse reported issues
        reported_keys: Set[str] = set()
        parse_errors: list[str] = []
        for raw_issue in action.issues:
            key = parse_issue_key(raw_issue)
            if key:
                reported_keys.add(key)
            else:
                parse_errors.append(f"Could not parse: '{raw_issue}'")

        # Compute score (standard F1)
        metrics = compute_f1(reported_keys, self._planted_keys)
        score = metrics["f1"]

        # Compute difficulty-weighted reward (richer per-step signal)
        weighted = compute_weighted_reward(reported_keys, self._current_task.planted_issues)
        weighted_reward = weighted["weighted_reward"]

        # Use weighted reward as the primary reward signal
        self._best_score = max(self._best_score, weighted_reward)
        self._state.best_score = self._best_score

        # Check if done
        is_done = (
            score >= 0.999  # Perfect score (all issues found exactly)
            or self._state.current_step >= self._state.max_steps
        )

        # Build feedback
        feedback_lines = [
            f"Step {self._state.current_step}/{self._state.max_steps}",
            f"Issues reported: {len(reported_keys)}",
            f"True positives: {metrics['tp']}, False positives: {metrics['fp']}, Missed: {metrics['fn']}",
            f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {score:.3f}",
            f"Weighted reward: {weighted_reward:.3f} (difficulty found: {weighted['difficulty_found']}, missed: {weighted['difficulty_missed']})",
        ]

        if parse_errors:
            feedback_lines.append(f"Parse errors ({len(parse_errors)}): {'; '.join(parse_errors[:3])}")

        if not is_done:
            # Give hints about what was missed without revealing exact answers
            if metrics["fn"] > 0:
                feedback_lines.append(
                    f"You missed {metrics['fn']} issue(s). Review the dataset carefully."
                )
            if metrics["fp"] > 0:
                feedback_lines.append(
                    f"{metrics['fp']} of your reported issues were incorrect."
                )
            feedback_lines.append("You can submit again with an updated list of issues.")
        else:
            feedback_lines.append(f"Task complete! Final best weighted reward: {self._best_score:.3f}")

        return DataQAObservation(
            dataset_csv=self._current_task.corrupted_csv,
            schema_description=self._current_task.schema_description,
            validation_rules=self._current_task.validation_rules,
            task_description=self._current_task.description,
            feedback="\n".join(feedback_lines),
            task_id=self._current_task.task_id,
            num_issues_hint=len(self._current_task.planted_issues),
            max_steps=self._state.max_steps,
            done=is_done,
            reward=self._best_score,
            metadata={
                "f1": score,
                "weighted_reward": weighted_reward,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "difficulty_found": weighted["difficulty_found"],
                "difficulty_missed": weighted["difficulty_missed"],
            },
        )

    @property
    def state(self) -> DataQAState:
        return self._state
