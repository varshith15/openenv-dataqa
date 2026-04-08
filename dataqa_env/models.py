"""
DataQA Environment Models
-------------------------
Action/Observation/State types for the Data Quality Assurance environment.

The agent receives a dataset with planted quality issues and must identify them.
Grading is based on F1 score (precision × recall) of correctly identified issues.
"""

from __future__ import annotations

from typing import List, Optional

from openenv.core.env_server.interfaces import Action, Observation, State


class DataQAAction(Action):
    """
    Agent submits identified issues AND optional proposed fixes.

    Two-phase action space:
      Phase 1 (Identify): List issues in format "row:<N>,col:<name>,issue:<type>"
      Phase 2 (Fix):      List fixes in format "row:<N>,col:<name>,fix:<proposed_value>"

    The agent can submit both in the same step or across multiple steps.
    Combined reward = 0.6 * identify_score + 0.4 * fix_score

    Supported issue types:
        missing_value, wrong_type, duplicate_row, out_of_range,
        format_violation, inconsistent_value, statistical_outlier,
        referential_integrity
    """

    issues: List[str]
    fixes: List[str] = []
    # Include task_id so step() can reconstruct context in stateless HTTP mode
    task_id: str = "easy"


class DataQAObservation(Observation):
    """
    What the agent sees: a dataset, its schema/rules, and feedback.
    """

    # The dataset as CSV text
    dataset_csv: str = ""

    # Schema description (column names, expected types, constraints)
    schema_description: str = ""

    # Validation rules in plain text
    validation_rules: str = ""

    # Task description
    task_description: str = ""

    # Feedback from previous step (empty on reset)
    feedback: str = ""

    # Current task ID
    task_id: str = ""

    # Number of planted issues (hint for the agent)
    num_issues_hint: int = 0

    # Max allowed steps for this task
    max_steps: int = 3


class DataQAState(State):
    """Tracks episode progress."""

    task_id: str = ""
    current_step: int = 0
    max_steps: int = 3
    best_score: float = 0.0
    total_planted_issues: int = 0
