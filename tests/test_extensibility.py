"""Tests for the extensibility API — custom tasks and contamination rules."""

import pytest
from dataqa_env.server.tasks import (
    PlantedIssue,
    create_task_from_config,
    register_task,
    register_contamination_rule,
    CONTAMINATION_RULES,
    get_task,
    list_tasks,
)
from dataqa_env.server.environment import DataQAEnvironment, compute_weighted_reward
from dataqa_env.models import DataQAAction


SIMPLE_CSV = "id,name,score\n1,Alice,95\n2,Bob,87\n3,Carol,92\n4,Dave,78"


class TestCreateTaskFromConfig:
    def test_basic_creation(self):
        task = create_task_from_config(
            task_id="test_custom",
            name="Test Task",
            description="Test",
            schema_description="id: int, name: str, score: int",
            validation_rules="No missing values",
            clean_csv=SIMPLE_CSV,
            contaminations=[
                {"rule": "missing_value", "row": 0, "col": 1},
            ],
        )
        assert task.task_id == "test_custom"
        assert len(task.planted_issues) == 1
        assert task.planted_issues[0].issue_type == "missing_value"
        assert task.planted_issues[0].col == "name"

    def test_multiple_contaminations(self):
        task = create_task_from_config(
            task_id="multi",
            name="Multi",
            description="Test",
            schema_description="",
            validation_rules="",
            clean_csv=SIMPLE_CSV,
            contaminations=[
                {"rule": "missing_value", "row": 0, "col": 1},
                {"rule": "missing_value", "row": 2, "col": 1},
            ],
        )
        assert len(task.planted_issues) == 2

    def test_custom_difficulty_override(self):
        task = create_task_from_config(
            task_id="custom_diff",
            name="Custom Difficulty",
            description="Test",
            schema_description="",
            validation_rules="",
            clean_csv=SIMPLE_CSV,
            contaminations=[
                {"rule": "missing_value", "row": 0, "col": 1, "difficulty": 2.5},
            ],
        )
        assert task.planted_issues[0].difficulty == 2.5

    def test_callable_rule(self):
        def custom_rule(rows, header, col_idx, row_idx, rng):
            return "CORRUPTED", PlantedIssue(
                row=row_idx + 1, col=header[col_idx], issue_type="wrong_type",
                description="Custom corruption", difficulty=1.5,
            )

        task = create_task_from_config(
            task_id="callable",
            name="Callable Rule",
            description="Test",
            schema_description="",
            validation_rules="",
            clean_csv=SIMPLE_CSV,
            contaminations=[
                {"rule": custom_rule, "row": 1, "col": 2},
            ],
        )
        assert task.planted_issues[0].issue_type == "wrong_type"
        assert "CORRUPTED" in task.corrupted_csv

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown contamination rule"):
            create_task_from_config(
                task_id="bad",
                name="Bad",
                description="",
                schema_description="",
                validation_rules="",
                clean_csv=SIMPLE_CSV,
                contaminations=[{"rule": "nonexistent_rule", "row": 0, "col": 0}],
            )


class TestRegisterContaminationRule:
    def test_register_and_use(self):
        def reverse_value(rows, header, col_idx, row_idx, rng):
            val = rows[row_idx][col_idx]
            return val[::-1], PlantedIssue(
                row=row_idx + 1, col=header[col_idx], issue_type="format_violation",
                description="Reversed value", difficulty=1.5,
            )

        register_contamination_rule("reverse", reverse_value)
        assert "reverse" in CONTAMINATION_RULES

        task = create_task_from_config(
            task_id="rev_test",
            name="Reverse Test",
            description="",
            schema_description="",
            validation_rules="",
            clean_csv=SIMPLE_CSV,
            contaminations=[{"rule": "reverse", "row": 0, "col": 1}],
        )
        assert task.planted_issues[0].issue_type == "format_violation"
        # "Alice" reversed is "ecilA"
        assert "ecilA" in task.corrupted_csv

        # Cleanup
        del CONTAMINATION_RULES["reverse"]


class TestRegisterTask:
    def test_register_and_get(self):
        task = create_task_from_config(
            task_id="registered",
            name="Registered Task",
            description="Test registered task",
            schema_description="id: int, name: str",
            validation_rules="No missing values",
            clean_csv=SIMPLE_CSV,
            contaminations=[{"rule": "missing_value", "row": 1, "col": 1}],
        )
        register_task("registered", lambda seed: task)
        assert "registered" in list_tasks()

        fetched = get_task("registered")
        assert fetched.task_id == "registered"
        assert len(fetched.planted_issues) == 1

        # Cleanup
        from dataqa_env.server.tasks import TASK_REGISTRY
        del TASK_REGISTRY["registered"]


class TestCustomTaskInEnvironment:
    def test_full_lifecycle_identify_only(self):
        """Custom task works end-to-end with identify-only."""
        task = create_task_from_config(
            task_id="e2e_custom",
            name="E2E Custom",
            description="End-to-end test",
            schema_description="id: int, name: str, score: int",
            validation_rules="No missing values",
            clean_csv=SIMPLE_CSV,
            contaminations=[
                {"rule": "missing_value", "row": 0, "col": 1, "difficulty": 1.0},
                {"rule": "whitespace_value", "row": 2, "col": 1, "difficulty": 2.5},
            ],
        )
        register_task("e2e_custom", lambda seed: task)

        env = DataQAEnvironment()
        obs = env.reset(task_id="e2e_custom")
        assert obs.num_issues_hint == 2

        action = DataQAAction(
            issues=[i.to_key() for i in task.planted_issues],
            task_id="e2e_custom",
        )
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward >= 0.999

        from dataqa_env.server.tasks import TASK_REGISTRY
        del TASK_REGISTRY["e2e_custom"]

    def test_full_lifecycle_identify_and_fix(self):
        """Custom task works end-to-end with both identify and fix."""
        task = create_task_from_config(
            task_id="e2e_fix",
            name="E2E Fix",
            description="End-to-end test with fixes",
            schema_description="id: int, name: str, score: int",
            validation_rules="No missing values",
            clean_csv=SIMPLE_CSV,
            contaminations=[
                {"rule": "missing_value", "row": 0, "col": 1, "difficulty": 1.0},
            ],
        )
        register_task("e2e_fix", lambda seed: task)

        env = DataQAEnvironment()
        env.reset(task_id="e2e_fix")

        # Submit issues + fix
        action = DataQAAction(
            issues=[task.planted_issues[0].to_key()],
            fixes=["row:1,col:name,fix:Alice"],  # clean value is "Alice"
            task_id="e2e_fix",
        )
        obs = env.step(action)
        assert obs.done is True
        assert obs.metadata["fix_score"] > 0.0
        assert obs.metadata["combined_reward"] > 0.0

        from dataqa_env.server.tasks import TASK_REGISTRY
        del TASK_REGISTRY["e2e_fix"]
