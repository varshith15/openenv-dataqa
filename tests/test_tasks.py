"""Tests for task definitions, data corruption, and issue planting."""

import pytest
from dataqa_env.server.tasks import (
    PlantedIssue,
    Task,
    create_task_easy,
    create_task_medium,
    create_task_hard,
    get_task,
    list_tasks,
    _csv_to_rows,
    _rows_to_csv,
)


class TestPlantedIssue:
    def test_to_key(self):
        issue = PlantedIssue(row=3, col="salary", issue_type="missing_value", description="test")
        assert issue.to_key() == "row:3,col:salary,issue:missing_value"

    def test_difficulty_default(self):
        issue = PlantedIssue(row=1, col="name", issue_type="missing_value", description="test")
        assert issue.difficulty == 1.0

    def test_difficulty_custom(self):
        issue = PlantedIssue(row=1, col="name", issue_type="missing_value", description="test", difficulty=3.0)
        assert issue.difficulty == 3.0


class TestCSVHelpers:
    def test_roundtrip(self):
        csv_text = "a,b,c\n1,2,3\n4,5,6"
        rows = _csv_to_rows(csv_text)
        assert len(rows) == 3
        result = _rows_to_csv(rows)
        assert "1,2,3" in result

    def test_empty_csv(self):
        rows = _csv_to_rows("a,b\n")
        assert len(rows) == 1  # header only


class TestTaskEasy:
    @pytest.fixture
    def task(self):
        return create_task_easy()

    def test_task_id(self, task):
        assert task.task_id == "easy"

    def test_has_6_issues(self, task):
        assert len(task.planted_issues) == 6

    def test_issue_types(self, task):
        types = {i.issue_type for i in task.planted_issues}
        assert "missing_value" in types
        assert "wrong_type" in types
        assert "duplicate_row" in types
        assert "out_of_range" in types
        assert "inconsistent_value" in types

    def test_corrupted_csv_differs_from_clean(self, task):
        assert task.corrupted_csv != task.clean_csv

    def test_issue_keys_unique(self, task):
        keys = [i.to_key() for i in task.planted_issues]
        assert len(keys) == len(set(keys))

    def test_max_steps(self, task):
        assert task.max_steps == 3

    def test_corrupted_csv_has_more_rows(self, task):
        clean_rows = _csv_to_rows(task.clean_csv)
        corrupt_rows = _csv_to_rows(task.corrupted_csv)
        assert len(corrupt_rows) > len(clean_rows)  # duplicate row added

    def test_difficulty_weights(self, task):
        for issue in task.planted_issues:
            assert 1.0 <= issue.difficulty <= 3.0


class TestTaskMedium:
    @pytest.fixture
    def task(self):
        return create_task_medium()

    def test_task_id(self, task):
        assert task.task_id == "medium"

    def test_has_8_issues(self, task):
        assert len(task.planted_issues) == 8

    def test_issue_types(self, task):
        types = {i.issue_type for i in task.planted_issues}
        assert "inconsistent_value" in types
        assert "format_violation" in types
        assert "wrong_type" in types

    def test_issue_keys_unique(self, task):
        keys = [i.to_key() for i in task.planted_issues]
        assert len(keys) == len(set(keys))

    def test_difficulty_weights(self, task):
        for issue in task.planted_issues:
            assert 1.0 <= issue.difficulty <= 3.0


class TestTaskHard:
    @pytest.fixture
    def task(self):
        return create_task_hard()

    def test_task_id(self, task):
        assert task.task_id == "hard"

    def test_has_10_issues(self, task):
        assert len(task.planted_issues) == 10

    def test_issue_types(self, task):
        types = {i.issue_type for i in task.planted_issues}
        assert "inconsistent_value" in types
        assert "format_violation" in types
        assert "statistical_outlier" in types
        assert "out_of_range" in types

    def test_has_high_difficulty_issues(self, task):
        hard_issues = [i for i in task.planted_issues if i.difficulty >= 2.5]
        assert len(hard_issues) >= 2  # data leakage, GPU outlier, whitespace

    def test_issue_keys_unique(self, task):
        keys = [i.to_key() for i in task.planted_issues]
        assert len(keys) == len(set(keys))


class TestTaskAlignment:
    @pytest.fixture
    def task(self):
        return create_task_hard()  # reuse import, we'll import alignment below

    def test_alignment_task(self):
        from dataqa_env.server.tasks import get_task
        task = get_task("alignment")
        assert task.task_id == "alignment"
        assert len(task.planted_issues) == 12

    def test_alignment_issue_types(self):
        from dataqa_env.server.tasks import get_task
        task = get_task("alignment")
        types = {i.issue_type for i in task.planted_issues}
        assert "inconsistent_value" in types  # factual errors, mismatches, hallucinations
        assert "missing_value" in types        # truncated, whitespace-only
        assert "duplicate_row" in types        # duplicate instruction

    def test_alignment_has_high_difficulty(self):
        from dataqa_env.server.tasks import get_task
        task = get_task("alignment")
        hard_issues = [i for i in task.planted_issues if i.difficulty >= 2.5]
        assert len(hard_issues) >= 3  # hallucinated citation, harmful advice, factual error

    def test_alignment_issue_keys_unique(self):
        from dataqa_env.server.tasks import get_task
        task = get_task("alignment")
        keys = [i.to_key() for i in task.planted_issues]
        assert len(keys) == len(set(keys))

    def test_alignment_corrupted_differs(self):
        from dataqa_env.server.tasks import get_task
        task = get_task("alignment")
        assert task.corrupted_csv != task.clean_csv

    def test_alignment_in_env(self):
        from dataqa_env.server.environment import DataQAEnvironment
        from dataqa_env.models import DataQAAction
        env = DataQAEnvironment()
        obs = env.reset(task_id="alignment")
        assert obs.num_issues_hint == 12
        # Perfect submission
        from dataqa_env.server.tasks import get_task
        task = get_task("alignment")
        action = DataQAAction(issues=[i.to_key() for i in task.planted_issues], task_id="alignment")
        obs = env.step(action)
        assert obs.reward >= 0.99


class TestTaskRegistry:
    def test_list_tasks(self):
        tasks = list_tasks()
        assert set(tasks) == {"easy", "medium", "hard", "alignment"}

    def test_get_task_easy(self):
        task = get_task("easy")
        assert task.task_id == "easy"

    def test_get_task_medium(self):
        task = get_task("medium")
        assert task.task_id == "medium"

    def test_get_task_hard(self):
        task = get_task("hard")
        assert task.task_id == "hard"

    def test_get_task_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent")

    def test_seed_determinism(self):
        t1 = get_task("easy", seed=42)
        t2 = get_task("easy", seed=42)
        assert t1.corrupted_csv == t2.corrupted_csv
        assert [i.to_key() for i in t1.planted_issues] == [i.to_key() for i in t2.planted_issues]
