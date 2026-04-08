"""Tests for the DataQA environment (reset, step, scoring, two-phase identify+fix)."""

import pytest
from dataqa_env.server.environment import (
    DataQAEnvironment,
    parse_issue_key,
    parse_fix,
    compute_f1,
    compute_weighted_reward,
    grade_fixes,
    IDENTIFY_WEIGHT,
    FIX_WEIGHT,
)
from dataqa_env.models import DataQAAction
from dataqa_env.server.tasks import PlantedIssue, create_task_easy, create_task_medium


# ──────────────────────────────────────────────────────
# Issue parsing
# ──────────────────────────────────────────────────────

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
        assert parse_issue_key("row:3,col:salary") is None

    def test_empty_string(self):
        assert parse_issue_key("") is None

    def test_semicolon_separator(self):
        result = parse_issue_key("row:3;col:salary;issue:missing_value")
        assert result == "row:3,col:salary,issue:missing_value"


# ──────────────────────────────────────────────────────
# Fix parsing
# ──────────────────────────────────────────────────────

class TestParseFix:
    def test_standard_format(self):
        result = parse_fix("row:4,col:name,fix:Alice Chen")
        assert result == (4, "name", "Alice Chen")

    def test_with_equals(self):
        result = parse_fix("row=4,col=name,fix=Alice Chen")
        assert result == (4, "name", "Alice Chen")

    def test_numeric_fix(self):
        result = parse_fix("row:7,col:salary,fix:75000")
        assert result == (7, "salary", "75000")

    def test_date_fix(self):
        result = parse_fix("row:12,col:order_date,fix:2024-01-26")
        assert result == (12, "order_date", "2024-01-26")

    def test_case_insensitive(self):
        result = parse_fix("Row:4,Col:Name,Fix:Alice Chen")
        assert result == (4, "name", "Alice Chen")

    def test_unparseable(self):
        assert parse_fix("garbage") is None
        assert parse_fix("row:4,col:name") is None

    def test_fix_with_special_chars(self):
        result = parse_fix("row:1,col:email,fix:alice.chen@company.com")
        assert result == (1, "email", "alice.chen@company.com")


# ──────────────────────────────────────────────────────
# F1 scoring
# ──────────────────────────────────────────────────────

class TestComputeF1:
    def test_perfect_match(self):
        keys = {"row:1,col:a,issue:missing_value"}
        result = compute_f1(keys, keys)
        assert result["f1"] == 1.0

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


# ──────────────────────────────────────────────────────
# Weighted reward
# ──────────────────────────────────────────────────────

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

    def test_hard_issue_worth_more(self):
        easy = PlantedIssue(row=1, col="a", issue_type="missing_value", description="", difficulty=1.0)
        hard = PlantedIssue(row=2, col="b", issue_type="statistical_outlier", description="", difficulty=3.0)
        issues = [easy, hard]
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


# ──────────────────────────────────────────────────────
# Fix grading
# ──────────────────────────────────────────────────────

class TestGradeFixes:
    @pytest.fixture
    def easy_task(self):
        return create_task_easy()

    def test_no_fixes_no_issues(self):
        from dataqa_env.server.tasks import Task
        task = Task(task_id="empty", name="", description="", schema_description="",
                    validation_rules="", clean_csv="a\n1")
        result = grade_fixes([], task)
        assert result["fix_score"] == 1.0

    def test_no_fixes_submitted(self, easy_task):
        result = grade_fixes([], easy_task)
        assert result["fix_score"] == 0.0
        assert result["fixes_attempted"] == 0

    def test_exact_fix_for_missing_name(self, easy_task):
        # Row 4 has empty name — clean value is "David Kim"
        fixes = [(4, "name", "David Kim")]
        result = grade_fixes(fixes, easy_task)
        assert result["fix_score"] > 0.0
        assert result["fixes_correct"] == 1

    def test_exact_fix_for_wrong_type_salary(self, easy_task):
        # Row 7 has "seventy-five thousand" — clean value is "75000"
        fixes = [(7, "salary", "75000")]
        result = grade_fixes(fixes, easy_task)
        assert result["fixes_correct"] == 1

    def test_misspelling_fix(self, easy_task):
        # Row 11 has department "Engneering" — fix to "Engineering"
        fixes = [(11, "department", "Engineering")]
        result = grade_fixes(fixes, easy_task)
        assert result["fixes_correct"] == 1

    def test_wrong_value_for_issue_cell(self, easy_task):
        # Row 4 name is empty — propose wrong name
        fixes = [(4, "name", "Wrong Person")]
        result = grade_fixes(fixes, easy_task)
        assert result["fixes_partial"] == 1  # correct cell, wrong value
        assert result["fix_score"] > 0.0  # gets partial credit

    def test_fix_for_non_issue_cell(self, easy_task):
        # Row 1 col name is fine — no issue there
        fixes = [(1, "name", "Some Name")]
        result = grade_fixes(fixes, easy_task)
        assert result["fixes_wrong"] == 1
        assert result["fix_score"] == 0.0

    def test_multiple_fixes_best_wins(self, easy_task):
        # Submit two fixes for same cell — best one should count
        fixes = [
            (4, "name", "Wrong Person"),   # partial credit
            (4, "name", "David Kim"),      # exact match
        ]
        result = grade_fixes(fixes, easy_task)
        assert result["fixes_correct"] >= 1

    def test_all_fixes_correct(self, easy_task):
        # Fix deterministic issues with exact values
        fixes = [
            (4, "name", "David Kim"),        # inferred from email
            (7, "salary", "75000"),           # type conversion
            (11, "department", "Engineering"), # spelling fix
            (15, "email", "oscar.rivera@company.com"),  # pattern match
            (18, "salary", "99000"),          # remove extra digit
        ]
        result = grade_fixes(fixes, easy_task)
        assert result["fix_score"] > 0.7

    def test_fix_score_bounded(self, easy_task):
        fixes = [(4, "name", "David Kim"), (99, "x", "bad")]
        result = grade_fixes(fixes, easy_task)
        assert 0.0 <= result["fix_score"] <= 1.0


# ──────────────────────────────────────────────────────
# Full environment lifecycle
# ──────────────────────────────────────────────────────

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
        assert obs.num_issues_hint == 6
        assert obs.max_steps == 3
        assert obs.done is False
        assert obs.reward == 0.0
        assert "fix" in obs.feedback.lower()  # mentions fix phase

    def test_reset_medium(self, env):
        obs = env.reset(task_id="medium")
        assert obs.num_issues_hint == 8

    def test_reset_hard(self, env):
        obs = env.reset(task_id="hard")
        assert obs.num_issues_hint == 10

    def test_step_identify_only(self, env):
        """Backward compatible: only issues, no fixes."""
        env.reset(task_id="easy")
        # Submit all 6 correct issues for easy task
        from dataqa_env.server.tasks import get_task
        task = get_task("easy")
        action = DataQAAction(
            issues=[i.to_key() for i in task.planted_issues],
            task_id="easy",
        )
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward >= 0.999

    def test_step_with_fixes_increases_reward(self, env):
        """Submitting correct fixes should produce high combined reward."""
        env.reset(task_id="easy")
        from dataqa_env.server.tasks import get_task
        task = get_task("easy")
        action = DataQAAction(
            issues=[i.to_key() for i in task.planted_issues],
            fixes=[
                "row:4,col:name,fix:David Kim",
                "row:7,col:salary,fix:75000",
                "row:9,col:department,fix:Engineering",
            ],
            task_id="easy",
        )
        obs = env.step(action)
        assert obs.metadata["combined_reward"] > 0.7

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

        action2 = DataQAAction(issues=["row:99,col:x,issue:wrong_type"], task_id="easy")
        env.step(action2)
        assert env.state.best_score >= score_after_1

    def test_metadata_includes_both_phases(self, env):
        env.reset(task_id="easy")
        action = DataQAAction(
            issues=["row:4,col:name,issue:missing_value"],
            fixes=["row:4,col:name,fix:David Kim"],
            task_id="easy",
        )
        obs = env.step(action)
        m = obs.metadata
        assert "identify_f1" in m
        assert "identify_score" in m
        assert "fix_score" in m
        assert "combined_reward" in m
        assert "tp" in m
        assert "fixes_correct" in m
        assert "fixes_attempted" in m

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
                fixes=["row:1,col:x,fix:wrong"],
                task_id="hard",
            )
            obs = env.step(action)
            assert 0.0 <= obs.reward <= 1.0

    def test_combined_reward_weights(self, env):
        """Verify combined = IDENTIFY_WEIGHT * identify + FIX_WEIGHT * fix."""
        env.reset(task_id="easy")
        action = DataQAAction(
            issues=["row:4,col:name,issue:missing_value"],
            fixes=["row:4,col:name,fix:David Kim"],
            task_id="easy",
        )
        obs = env.step(action)
        m = obs.metadata
        expected = IDENTIFY_WEIGHT * m["identify_score"] + FIX_WEIGHT * m["fix_score"]
        assert abs(m["combined_reward"] - expected) < 0.01

    def test_fix_feedback_shown_when_fixes_submitted(self, env):
        env.reset(task_id="easy")
        action = DataQAAction(
            issues=["row:4,col:name,issue:missing_value"],
            fixes=["row:4,col:name,fix:David Kim"],
            task_id="easy",
        )
        obs = env.step(action)
        assert "Fix Proposals" in obs.feedback
        assert "Combined Reward" in obs.feedback

    def test_no_fix_penalty_when_no_fixes_submitted(self, env):
        """If agent submits no fixes, reward = identify_score (no penalty)."""
        env.reset(task_id="easy")
        from dataqa_env.server.tasks import get_task
        task = get_task("easy")
        action = DataQAAction(
            issues=[i.to_key() for i in task.planted_issues],
            task_id="easy",
        )
        obs = env.step(action)
        assert obs.reward >= 0.99
        assert obs.metadata["combined_reward"] == obs.metadata["identify_score"]
