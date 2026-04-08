"""Tests for the DataQA environment (reset, step, scoring)."""

import pytest
from dataqa_env.server.environment import (
    DataQAEnvironment,
    parse_issue_key,
    compute_f1,
    compute_weighted_reward,
)
from dataqa_env.models import DataQAAction
from dataqa_env.server.tasks import PlantedIssue


class TestParseIssueKey:
    def test_standard_format(self):
        assert parse_issue_key("row:3,col:salary,issue:missing_value") == "row:3,col:salary,issue:missing_value"

    def test_with_equals(self):
        assert parse_issue_key("row=3,col=salary,issue=missing_value") == "row:3,col:salary,issue:missing_value"

    def test_case_insensitive(self):
        assert parse_issue_key("Row:3,Col:Salary,Issue:Missing_Value") == "row:3,col:salary,issue:missing_value"

    def test_with_spaces(self):
        assert parse_issue_key("row: 3, col: salary, issue: missing_value") == "row:3,col:salary,issue:missing_value"

    def test_unparseable(self):
        assert parse_issue_key("this is garbage") is None

    def test_partial_match(self):
        assert parse_issue_key("row:3,col:salary") is None  # missing issue

    def test_empty_string(self):
        assert parse_issue_key("") is None

    def test_semicolon_separator(self):
        result = parse_issue_key("row:3;col:salary;issue:missing_value")
        assert result == "row:3,col:salary,issue:missing_value"


class TestComputeF1:
    def test_perfect_match(self):
        keys = {"row:1,col:a,issue:missing_value"}
        result = compute_f1(keys, keys)
        assert result["f1"] == 1.0
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_no_reported_no_planted(self):
        result = compute_f1(set(), set())
        assert result["f1"] == 1.0

    def test_no_reported_some_planted(self):
        planted = {"row:1,col:a,issue:missing_value"}
        result = compute_f1(set(), planted)
        assert result["f1"] == 0.0
        assert result["fn"] == 1

    def test_all_false_positives(self):
        reported = {"row:99,col:x,issue:wrong_type"}
        planted = {"row:1,col:a,issue:missing_value"}
        result = compute_f1(reported, planted)
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1
        assert result["f1"] == 0.0

    def test_partial_match(self):
        reported = {"row:1,col:a,issue:missing_value", "row:2,col:b,issue:wrong_type"}
        planted = {"row:1,col:a,issue:missing_value", "row:3,col:c,issue:duplicate_row"}
        result = compute_f1(reported, planted)
        assert result["tp"] == 1
        assert result["fp"] == 1
        assert result["fn"] == 1
        assert 0 < result["f1"] < 1

    def test_precision_recall_calculation(self):
        reported = {"a", "b", "c"}
        planted = {"a", "b", "d"}
        result = compute_f1(reported, planted)
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(2 / 3)


class TestComputeWeightedReward:
    def test_perfect_match(self):
        issues = [
            PlantedIssue(row=1, col="a", issue_type="missing_value", description="", difficulty=1.0),
            PlantedIssue(row=2, col="b", issue_type="wrong_type", description="", difficulty=3.0),
        ]
        reported = {i.to_key() for i in issues}
        result = compute_weighted_reward(reported, issues)
        assert result["weighted_reward"] == 1.0

    def test_empty_both(self):
        result = compute_weighted_reward(set(), [])
        assert result["weighted_reward"] == 1.0

    def test_no_reported(self):
        issues = [PlantedIssue(row=1, col="a", issue_type="missing_value", description="", difficulty=2.0)]
        result = compute_weighted_reward(set(), issues)
        assert result["weighted_reward"] == 0.0
        assert result["difficulty_missed"] == 2.0

    def test_hard_issue_worth_more(self):
        easy = PlantedIssue(row=1, col="a", issue_type="missing_value", description="", difficulty=1.0)
        hard = PlantedIssue(row=2, col="b", issue_type="statistical_outlier", description="", difficulty=3.0)
        issues = [easy, hard]

        # Finding only the hard issue should score higher than only the easy issue
        hard_found = compute_weighted_reward({hard.to_key()}, issues)
        easy_found = compute_weighted_reward({easy.to_key()}, issues)
        assert hard_found["weighted_reward"] > easy_found["weighted_reward"]

    def test_false_positives_reduce_reward(self):
        issues = [PlantedIssue(row=1, col="a", issue_type="missing_value", description="", difficulty=1.0)]
        correct = {issues[0].to_key()}
        with_fp = correct | {"row:99,col:x,issue:wrong_type"}
        r_correct = compute_weighted_reward(correct, issues)
        r_with_fp = compute_weighted_reward(with_fp, issues)
        assert r_correct["weighted_reward"] > r_with_fp["weighted_reward"]


class TestDataQAEnvironment:
    @pytest.fixture
    def env(self):
        return DataQAEnvironment()

    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="easy")
        assert obs.dataset_csv
        assert obs.schema_description
        assert obs.validation_rules
        assert obs.task_description
        assert obs.num_issues_hint == 4
        assert obs.max_steps == 3
        assert obs.done is False
        assert obs.reward == 0.0

    def test_reset_medium(self, env):
        obs = env.reset(task_id="medium")
        assert obs.num_issues_hint == 6

    def test_reset_hard(self, env):
        obs = env.reset(task_id="hard")
        assert obs.num_issues_hint == 8

    def test_step_with_correct_issues(self, env):
        env.reset(task_id="easy")
        # Submit all correct issues for easy task
        action = DataQAAction(
            issues=[
                "row:4,col:name,issue:missing_value",
                "row:7,col:salary,issue:wrong_type",
                "row:11,col:employee_id,issue:duplicate_row",
                "row:9,col:salary,issue:out_of_range",
            ],
            task_id="easy",
        )
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward >= 0.999

    def test_step_with_partial_issues(self, env):
        env.reset(task_id="easy")
        action = DataQAAction(
            issues=["row:4,col:name,issue:missing_value"],
            task_id="easy",
        )
        obs = env.step(action)
        assert 0 < obs.reward < 1.0
        assert obs.done is False

    def test_step_with_no_issues(self, env):
        env.reset(task_id="easy")
        action = DataQAAction(issues=[], task_id="easy")
        obs = env.step(action)
        assert obs.reward == 0.0

    def test_step_exhausts_max_steps(self, env):
        env.reset(task_id="easy")
        for _ in range(3):
            action = DataQAAction(issues=["row:99,col:x,issue:wrong_type"], task_id="easy")
            obs = env.step(action)
        assert obs.done is True

    def test_auto_reset_on_step(self, env):
        # step() without prior reset should auto-reset
        action = DataQAAction(
            issues=["row:4,col:name,issue:missing_value"],
            task_id="easy",
        )
        obs = env.step(action)
        assert obs.task_id == "easy"

    def test_state_tracking(self, env):
        env.reset(task_id="easy")
        assert env.state.task_id == "easy"
        assert env.state.current_step == 0
        assert env.state.best_score == 0.0

        action = DataQAAction(issues=["row:4,col:name,issue:missing_value"], task_id="easy")
        env.step(action)
        assert env.state.current_step == 1
        assert env.state.best_score > 0.0

    def test_best_score_monotonic(self, env):
        env.reset(task_id="easy")
        action1 = DataQAAction(
            issues=["row:4,col:name,issue:missing_value", "row:7,col:salary,issue:wrong_type"],
            task_id="easy",
        )
        env.step(action1)
        score_after_1 = env.state.best_score

        # Worse submission shouldn't decrease best_score
        action2 = DataQAAction(issues=["row:99,col:x,issue:wrong_type"], task_id="easy")
        env.step(action2)
        assert env.state.best_score >= score_after_1

    def test_metadata_included_in_observation(self, env):
        env.reset(task_id="easy")
        action = DataQAAction(issues=["row:4,col:name,issue:missing_value"], task_id="easy")
        obs = env.step(action)
        assert "f1" in obs.metadata
        assert "weighted_reward" in obs.metadata
        assert "tp" in obs.metadata
        assert "difficulty_found" in obs.metadata

    def test_parse_error_in_feedback(self, env):
        env.reset(task_id="easy")
        action = DataQAAction(issues=["garbage input"], task_id="easy")
        obs = env.step(action)
        assert "Parse error" in obs.feedback

    def test_concurrent_sessions_flag(self):
        assert DataQAEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True

    def test_reward_between_0_and_1(self, env):
        """Hackathon requirement: scores must be 0.0-1.0."""
        env.reset(task_id="hard")
        for _ in range(3):
            action = DataQAAction(
                issues=["row:1,col:x,issue:wrong_type", "row:99,col:y,issue:missing_value"],
                task_id="hard",
            )
            obs = env.step(action)
            assert 0.0 <= obs.reward <= 1.0
