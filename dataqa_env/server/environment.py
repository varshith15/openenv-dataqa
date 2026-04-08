"""
DataQA Environment
------------------
Server-side environment for data quality assurance tasks.

Two-phase RL environment:
  Phase 1 (Identify): Agent inspects corrupted datasets and reports quality issues.
  Phase 2 (Fix):      Agent proposes corrections for identified issues.

Combined reward = 0.6 * identify_score + 0.4 * fix_score
Both phases scored with difficulty-weighted metrics for rich per-step signal.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional, Set

from openenv.core.env_server.interfaces import Action, Environment, Observation

from ..models import DataQAAction, DataQAObservation, DataQAState
from .tasks import PlantedIssue, Task, get_task, list_tasks

# Reward weights for the two phases
IDENTIFY_WEIGHT = 0.6
FIX_WEIGHT = 0.4


def parse_issue_key(raw: str) -> Optional[str]:
    """
    Parse an agent-reported issue string into a normalized key.
    Expected format: row:<N>,col:<name>,issue:<type>
    Returns normalized key or None if unparseable.
    """
    raw = raw.strip().lower()
    row_match = re.search(r"row\s*[:=]\s*(\d+)", raw)
    col_match = re.search(r"col\s*[:=]\s*([\w_]+)", raw)
    issue_match = re.search(r"issue\s*[:=]\s*([\w_]+)", raw)

    if row_match and col_match and issue_match:
        return f"row:{row_match.group(1)},col:{col_match.group(1)},issue:{issue_match.group(1)}"
    return None


def parse_fix(raw: str) -> Optional[tuple[int, str, str]]:
    """
    Parse an agent-proposed fix into (row, col, proposed_value).
    Expected format: row:<N>,col:<name>,fix:<value>
    Returns (row, col, value) or None if unparseable.
    """
    raw = raw.strip()
    row_match = re.search(r"row\s*[:=]\s*(\d+)", raw, re.IGNORECASE)
    col_match = re.search(r"col(?:umn)?\s*[:=]\s*([\w_]+)", raw, re.IGNORECASE)
    fix_match = re.search(r"fix\s*[:=]\s*(.+?)$", raw, re.IGNORECASE)

    if row_match and col_match and fix_match:
        return (int(row_match.group(1)), col_match.group(1).lower(), fix_match.group(1).strip())
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

    found_keys = reported_keys & planted_keys
    missed_keys = planted_keys - reported_keys
    false_positive_count = len(reported_keys - planted_keys)

    difficulty_found = sum(planted_by_key[k].difficulty for k in found_keys)
    difficulty_missed = sum(planted_by_key[k].difficulty for k in missed_keys)
    total_weight = sum(i.difficulty for i in planted_issues)

    weighted_recall = difficulty_found / total_weight if total_weight > 0 else 0.0

    avg_difficulty = total_weight / len(planted_issues)
    fp_penalty_weight = false_positive_count * avg_difficulty
    weighted_precision = difficulty_found / (difficulty_found + fp_penalty_weight) if (difficulty_found + fp_penalty_weight) > 0 else 0.0

    if (weighted_precision + weighted_recall) > 0:
        weighted_reward = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
    else:
        weighted_reward = 0.0

    return {
        "weighted_reward": round(weighted_reward, 4),
        "difficulty_found": round(difficulty_found, 2),
        "difficulty_missed": round(difficulty_missed, 2),
    }


def grade_fixes(
    fixes: list[tuple[int, str, str]],
    task: Task,
) -> dict:
    """
    Grade proposed fixes against the clean dataset.

    For each fix (row, col, proposed_value), compare to the original clean value.
    Scoring per fix:
      - Exact match (case-insensitive, whitespace-stripped): 1.0
      - Numeric close match (within 1%): 0.8
      - Correct column but wrong value: 0.1
      - Targets a non-issue cell: 0.0 (penalty)

    Returns dict with fix_score (0.0-1.0), details per fix, and counts.
    """
    if not fixes and not task.planted_issues:
        return {"fix_score": 1.0, "fixes_correct": 0, "fixes_partial": 0,
                "fixes_wrong": 0, "fixes_attempted": 0, "fix_details": []}

    if not fixes:
        return {"fix_score": 0.0, "fixes_correct": 0, "fixes_partial": 0,
                "fixes_wrong": 0, "fixes_attempted": 0, "fix_details": []}

    issue_map = task.get_planted_issue_map()
    # Build set of (row, col) that are actual issues
    issue_cells = {(issue.row, issue.col) for issue in task.planted_issues}

    total_weight = sum(i.difficulty for i in task.planted_issues) if task.planted_issues else 1.0
    earned_weight = 0.0
    fixes_correct = 0
    fixes_partial = 0
    fixes_wrong = 0
    fix_details = []

    # Track which issues have been fixed (best fix wins)
    fixed_issues: dict[tuple[int, str], float] = {}

    for row, col, proposed in fixes:
        clean_value = task.get_clean_value(row, col)
        cell_key = (row, col)

        if cell_key not in issue_cells:
            # Fix targets a non-issue cell — no credit
            fix_details.append({"row": row, "col": col, "score": 0.0, "reason": "not an issue cell"})
            fixes_wrong += 1
            continue

        if clean_value is None:
            fix_details.append({"row": row, "col": col, "score": 0.0, "reason": "cell not found"})
            fixes_wrong += 1
            continue

        # Find the planted issue for this cell to get its difficulty weight
        matching_issue = None
        for issue in task.planted_issues:
            if issue.row == row and issue.col == col:
                matching_issue = issue
                break

        difficulty = matching_issue.difficulty if matching_issue else 1.0

        # Score the fix using tiered grading:
        #   1.0 = exact match with clean value
        #   0.8 = valid fix (right type, in range, addresses the issue) but not exact
        #   0.4 = partially valid (reasonable attempt, right direction)
        #   0.1 = targets correct cell but fix doesn't address the issue
        #   0.0 = makes things worse or targets non-issue cell
        score = 0.0
        reason = "wrong value"
        issue_type = matching_issue.issue_type if matching_issue else ""

        # Exact match (case-insensitive, whitespace-stripped)
        if proposed.strip().lower() == clean_value.lower():
            score = 1.0
            reason = "exact match"
            fixes_correct += 1
        else:
            # Grade by issue type — check if the fix is VALID even if not exact
            proposed_stripped = proposed.strip()

            if issue_type == "missing_value":
                # Any non-empty value is a reasonable fix for a missing value
                if proposed_stripped and proposed_stripped != " ":
                    score = 0.8
                    reason = "valid fix (non-empty value for missing field)"
                    fixes_partial += 1
                else:
                    score = 0.0
                    reason = "fix is still empty"
                    fixes_wrong += 1

            elif issue_type == "wrong_type":
                # Check if the proposed value is the correct type
                try:
                    float(proposed_stripped)
                    # Original was text, proposed is numeric — correct type fix
                    score = 0.8
                    reason = "valid fix (correct type)"
                    fixes_partial += 1
                except ValueError:
                    score = 0.1
                    reason = "fix is still wrong type"
                    fixes_partial += 1

            elif issue_type == "out_of_range":
                # Check if proposed value is within a reasonable range
                try:
                    proposed_num = float(proposed_stripped)
                    clean_num = float(clean_value)
                    # Within 50% of clean value = good estimate
                    if clean_num != 0 and abs(proposed_num - clean_num) / abs(clean_num) <= 0.5:
                        score = 0.8
                        reason = "valid fix (in reasonable range)"
                        fixes_partial += 1
                    elif proposed_num > 0 and (clean_num > 0) == (proposed_num > 0):
                        # At least right sign/direction
                        score = 0.4
                        reason = "partially valid (right direction)"
                        fixes_partial += 1
                    else:
                        score = 0.1
                        reason = "fix still out of reasonable range"
                        fixes_partial += 1
                except ValueError:
                    score = 0.1
                    reason = "correct cell, wrong value"
                    fixes_partial += 1

            elif issue_type == "format_violation":
                # Check if proposed value matches expected format
                # For dates: YYYY-MM-DD pattern
                if re.match(r"\d{4}-\d{2}-\d{2}", proposed_stripped):
                    score = 0.8
                    reason = "valid fix (correct format)"
                    fixes_partial += 1
                elif proposed_stripped and proposed_stripped != clean_value:
                    score = 0.4
                    reason = "fix attempted but format unclear"
                    fixes_partial += 1
                else:
                    score = 0.1
                    reason = "correct cell, wrong value"
                    fixes_partial += 1

            elif issue_type in ("inconsistent_value", "statistical_outlier"):
                # These require domain knowledge — any reasonable attempt gets partial credit
                try:
                    proposed_num = float(proposed_stripped)
                    clean_num = float(clean_value)
                    # Within 20% = strong fix, within 50% = reasonable
                    if clean_num != 0:
                        pct_diff = abs(proposed_num - clean_num) / abs(clean_num)
                        if pct_diff <= 0.01:
                            score = 1.0
                            reason = "exact numeric match"
                            fixes_correct += 1
                        elif pct_diff <= 0.2:
                            score = 0.8
                            reason = "valid fix (within 20% of correct value)"
                            fixes_partial += 1
                        elif pct_diff <= 0.5:
                            score = 0.4
                            reason = "partially valid (right ballpark)"
                            fixes_partial += 1
                        else:
                            score = 0.1
                            reason = "correct cell, value not close"
                            fixes_partial += 1
                    else:
                        score = 0.4
                        reason = "numeric fix attempted"
                        fixes_partial += 1
                except ValueError:
                    # Non-numeric fix for text fields — check similarity
                    if len(proposed_stripped) > 10 and proposed_stripped != clean_value:
                        score = 0.4
                        reason = "text fix attempted (cannot verify automatically)"
                        fixes_partial += 1
                    else:
                        score = 0.1
                        reason = "correct cell, wrong value"
                        fixes_partial += 1

            else:
                # Fallback: numeric close match or partial credit
                try:
                    proposed_num = float(proposed_stripped)
                    clean_num = float(clean_value)
                    if clean_num != 0 and abs(proposed_num - clean_num) / abs(clean_num) <= 0.01:
                        score = 0.8
                        reason = "numeric close match"
                        fixes_partial += 1
                    else:
                        score = 0.1
                        reason = "correct cell, wrong value"
                        fixes_partial += 1
                except (ValueError, ZeroDivisionError):
                    score = 0.1
                    reason = "correct cell, wrong value"
                    fixes_partial += 1

        # Keep best fix per cell
        if cell_key not in fixed_issues or score > fixed_issues[cell_key]:
            fixed_issues[cell_key] = score

        fix_details.append({"row": row, "col": col, "score": score, "reason": reason})

    # Compute fix score: weighted sum of best fix per issue / total weight
    for issue in task.planted_issues:
        cell_key = (issue.row, issue.col)
        if cell_key in fixed_issues:
            earned_weight += issue.difficulty * fixed_issues[cell_key]

    fix_score = earned_weight / total_weight if total_weight > 0 else 0.0
    fix_score = min(max(fix_score, 0.0), 1.0)

    return {
        "fix_score": round(fix_score, 4),
        "fixes_correct": fixes_correct,
        "fixes_partial": fixes_partial,
        "fixes_wrong": fixes_wrong,
        "fixes_attempted": len(fixes),
        "fix_details": fix_details,
    }


class DataQAEnvironment(Environment):
    """
    Data Quality Assurance environment — two-phase identify + fix.

    Phase 1 (Identify): Agent inspects corrupted datasets and reports quality issues.
    Phase 2 (Fix):      Agent proposes corrections for identified issues.

    Combined reward = 0.6 * identify_score + 0.4 * fix_score
    Both phases use difficulty-weighted scoring for rich per-step reward signals.
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
            feedback=(
                "Environment reset. Inspect the dataset and report all quality issues.\n"
                "You can also propose fixes in format: row:<N>,col:<name>,fix:<corrected_value>\n"
                "Combined reward = 0.6 * identify_score + 0.4 * fix_score"
            ),
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

        # Auto-reset in stateless HTTP mode
        if self._current_task is None:
            self.reset(task_id=action.task_id)

        self._state.step_count += 1
        self._state.current_step += 1

        # ── Phase 1: Parse and score issue identification ──
        reported_keys: Set[str] = set()
        parse_errors: list[str] = []
        for raw_issue in action.issues:
            key = parse_issue_key(raw_issue)
            if key:
                reported_keys.add(key)
            else:
                parse_errors.append(f"Could not parse issue: '{raw_issue}'")

        metrics = compute_f1(reported_keys, self._planted_keys)
        identify_f1 = metrics["f1"]

        weighted = compute_weighted_reward(reported_keys, self._current_task.planted_issues)
        identify_score = weighted["weighted_reward"]

        # ── Phase 2: Parse and score proposed fixes ──
        parsed_fixes: list[tuple[int, str, str]] = []
        for raw_fix in action.fixes:
            fix = parse_fix(raw_fix)
            if fix:
                parsed_fixes.append(fix)
            else:
                parse_errors.append(f"Could not parse fix: '{raw_fix}'")

        fix_result = grade_fixes(parsed_fixes, self._current_task)
        fix_score = fix_result["fix_score"]

        # ── Combined reward ──
        # If no fixes submitted, score is identify-only (no penalty for not fixing)
        if action.fixes:
            combined_reward = IDENTIFY_WEIGHT * identify_score + FIX_WEIGHT * fix_score
        else:
            combined_reward = identify_score  # backward compatible

        self._best_score = max(self._best_score, combined_reward)
        self._state.best_score = self._best_score

        # ── Check if done ──
        is_done = (
            identify_f1 >= 0.999  # Perfect identification
            or self._state.current_step >= self._state.max_steps
        )

        # ── Build feedback with actionable diagnostics ──
        # Show the agent exactly which reported issues were correct (TP) and which were wrong (FP)
        tp_keys = reported_keys & self._planted_keys
        fp_keys = reported_keys - self._planted_keys

        feedback_lines = [
            f"Step {self._state.current_step}/{self._state.max_steps}",
            "",
            "--- Identification ---",
            f"Issues reported: {len(reported_keys)}",
            f"True positives: {metrics['tp']}, False positives: {metrics['fp']}, Missed: {metrics['fn']}",
            f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {identify_f1:.3f}",
            f"Identify score (weighted): {identify_score:.3f}",
        ]

        # Show which reported issues were correct vs wrong (helps agent self-correct)
        if tp_keys:
            feedback_lines.append(f"Correct issues: {', '.join(sorted(tp_keys))}")
        if fp_keys:
            feedback_lines.append(f"Incorrect issues (false positives): {', '.join(sorted(fp_keys))}")

        if action.fixes:
            feedback_lines += [
                "",
                "--- Fix Proposals ---",
                f"Fixes attempted: {fix_result['fixes_attempted']}",
                f"Correct: {fix_result['fixes_correct']}, Partial: {fix_result['fixes_partial']}, Wrong: {fix_result['fixes_wrong']}",
                f"Fix score: {fix_score:.3f}",
            ]
            # Show per-fix feedback so agent knows which fixes worked
            for detail in fix_result["fix_details"]:
                status = "correct" if detail["score"] >= 0.99 else ("partial" if detail["score"] > 0 else "wrong")
                feedback_lines.append(
                    f"  row:{detail['row']},col:{detail['col']} -> {status} ({detail['reason']})"
                )
            feedback_lines.append(
                f"\n--- Combined Reward: {combined_reward:.3f} (identify={identify_score:.3f} x {IDENTIFY_WEIGHT} + fix={fix_score:.3f} x {FIX_WEIGHT}) ---"
            )
        else:
            feedback_lines += [
                "",
                "Tip: Submit fixes with format row:<N>,col:<name>,fix:<value> for bonus reward.",
            ]

        if parse_errors:
            feedback_lines.append(f"\nParse errors ({len(parse_errors)}): {'; '.join(parse_errors[:5])}")

        if not is_done:
            if metrics["fn"] > 0:
                feedback_lines.append(
                    f"\nYou missed {metrics['fn']} issue(s). Review the dataset carefully."
                )
            if metrics["fp"] > 0:
                feedback_lines.append(
                    f"Remove the {metrics['fp']} false positive(s) listed above and look for real issues."
                )
            feedback_lines.append("You can submit again with updated issues and/or fixes.")
        else:
            feedback_lines.append(f"\nTask complete! Final best reward: {self._best_score:.3f}")

        # ── Flag items for human review ──
        # In a production data QA pipeline, these would go to a human reviewer.
        # The grader flags cases where automated scoring has low confidence.
        human_review_flags: list[dict] = []

        # 1. False positives that target real columns — could be legitimate issues
        #    the task designer didn't plant (agent may be smarter than the grader)
        issue_map = self._current_task.get_planted_issue_map()
        valid_issue_types = {"missing_value", "wrong_type", "duplicate_row", "out_of_range",
                             "format_violation", "inconsistent_value", "statistical_outlier",
                             "referential_integrity"}
        for fp_key in fp_keys:
            parts = fp_key.split(",")
            itype = parts[2].split(":")[1] if len(parts) >= 3 else ""
            if itype in valid_issue_types:
                human_review_flags.append({
                    "item": fp_key,
                    "reason": "Agent reported this issue but it's not in ground truth — may be a real issue the grader missed",
                    "type": "possible_unplanted_issue",
                })

        # 2. Partial fix matches — fix was close but not exact, human should verify
        for detail in fix_result["fix_details"]:
            if 0 < detail["score"] < 0.99:
                human_review_flags.append({
                    "item": f"row:{detail['row']},col:{detail['col']}",
                    "reason": f"Fix scored {detail['score']:.2f} ({detail['reason']}) — human should verify if acceptable",
                    "type": "partial_fix",
                })

        # 3. High-difficulty issues that were missed — flag for training data review
        planted_by_key = {i.to_key(): i for i in self._current_task.planted_issues}
        fn_keys = self._planted_keys - reported_keys
        for fn_key in fn_keys:
            issue = planted_by_key.get(fn_key)
            if issue and issue.difficulty >= 2.5:
                human_review_flags.append({
                    "item": fn_key,
                    "reason": f"High-difficulty issue (difficulty={issue.difficulty}) missed — {issue.description}",
                    "type": "missed_hard_issue",
                })

        if human_review_flags:
            feedback_lines.append(f"\n--- Flagged for Human Review ({len(human_review_flags)}) ---")
            for flag in human_review_flags:
                feedback_lines.append(f"  [{flag['type']}] {flag['item']}: {flag['reason']}")

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
                "identify_f1": identify_f1,
                "identify_score": identify_score,
                "fix_score": fix_score,
                "combined_reward": combined_reward,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "difficulty_found": weighted["difficulty_found"],
                "difficulty_missed": weighted["difficulty_missed"],
                "fixes_correct": fix_result["fixes_correct"],
                "fixes_partial": fix_result["fixes_partial"],
                "fixes_wrong": fix_result["fixes_wrong"],
                "fixes_attempted": fix_result["fixes_attempted"],
                "fix_details": fix_result["fix_details"],
                "human_review_flags": human_review_flags,
            },
        )

    @property
    def state(self) -> DataQAState:
        return self._state
