"""Tests for the inference script's parsing, prompt building, and log format."""

import re
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference import parse_llm_response, build_user_prompt, log_start, log_step, log_end


class TestParseLLMResponse:
    def test_standard_format(self):
        response = "row:1,col:name,issue:missing_value\nrow:2,col:salary,issue:wrong_type"
        issues = parse_llm_response(response)
        assert len(issues) == 2
        assert "row:1,col:name,issue:missing_value" in issues

    def test_numbered_list(self):
        response = "1. row:1,col:name,issue:missing_value\n2. row:2,col:salary,issue:wrong_type"
        issues = parse_llm_response(response)
        assert len(issues) == 2

    def test_bullet_list(self):
        response = "- row:1,col:name,issue:missing_value\n* row:2,col:salary,issue:wrong_type"
        issues = parse_llm_response(response)
        assert len(issues) == 2

    def test_equals_delimiter(self):
        response = "row=1,col=name,issue=missing_value"
        issues = parse_llm_response(response)
        assert len(issues) == 1
        assert issues[0] == "row:1,col:name,issue:missing_value"

    def test_mixed_case(self):
        response = "Row:1,Col:Name,Issue:Missing_Value"
        issues = parse_llm_response(response)
        assert len(issues) == 1
        assert issues[0] == "row:1,col:name,issue:missing_value"

    def test_empty_response(self):
        assert parse_llm_response("") == []
        assert parse_llm_response("   ") == []

    def test_garbage_lines_skipped(self):
        response = "Here are the issues:\nrow:1,col:name,issue:missing_value\nNo more issues."
        issues = parse_llm_response(response)
        assert len(issues) == 1

    def test_deduplication_not_applied(self):
        response = "row:1,col:name,issue:missing_value\nrow:1,col:name,issue:missing_value"
        issues = parse_llm_response(response)
        assert len(issues) == 2  # duplicates kept, env handles dedup

    def test_with_column_variant(self):
        response = "row:1,column:name,issue:missing_value"
        issues = parse_llm_response(response)
        assert len(issues) == 1


class TestBuildUserPrompt:
    def test_includes_all_fields(self):
        obs = {
            "task_description": "Find issues",
            "schema_description": "col: int",
            "validation_rules": "no nulls",
            "dataset_csv": "a,b\n1,2",
            "num_issues_hint": 3,
            "feedback": "",
        }
        prompt = build_user_prompt(obs)
        assert "Find issues" in prompt
        assert "col: int" in prompt
        assert "no nulls" in prompt
        assert "a,b" in prompt
        assert "3 issues" in prompt

    def test_includes_feedback_on_retry(self):
        obs = {
            "task_description": "Find issues",
            "schema_description": "",
            "validation_rules": "",
            "dataset_csv": "a\n1",
            "num_issues_hint": 0,
            "feedback": "Step 1/3: You missed 2 issues",
        }
        prompt = build_user_prompt(obs)
        assert "FEEDBACK" in prompt
        assert "missed 2" in prompt

    def test_excludes_reset_feedback(self):
        obs = {
            "task_description": "",
            "schema_description": "",
            "validation_rules": "",
            "dataset_csv": "",
            "num_issues_hint": 0,
            "feedback": "Environment reset. Start inspecting.",
        }
        prompt = build_user_prompt(obs)
        assert "FEEDBACK" not in prompt


class TestLogFormat:
    """Verify stdout log format matches hackathon evaluation requirements."""

    def test_log_start_format(self, capsys):
        log_start(task="easy", env="dataqa_env", model="test-model")
        out = capsys.readouterr().out.strip()
        assert out == "[START] task=easy env=dataqa_env model=test-model"

    def test_log_step_format(self, capsys):
        log_step(step=1, action="row:1,col:name,issue:missing_value", reward=0.50, done=False, error=None)
        out = capsys.readouterr().out.strip()
        assert out == "[STEP] step=1 action=row:1,col:name,issue:missing_value reward=0.50 done=false error=null"

    def test_log_step_with_error(self, capsys):
        log_step(step=2, action="none", reward=0.00, done=True, error="timeout")
        out = capsys.readouterr().out.strip()
        assert "error=timeout" in out
        assert "done=true" in out

    def test_log_end_format(self, capsys):
        log_end(success=True, steps=3, score=0.85, rewards=[0.25, 0.50, 0.85])
        out = capsys.readouterr().out.strip()
        assert out == "[END] success=true steps=3 score=0.850 rewards=0.25,0.50,0.85"

    def test_log_end_failure(self, capsys):
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        out = capsys.readouterr().out.strip()
        assert "success=false" in out
        assert "score=0.000" in out

    def test_reward_format_2_decimal(self, capsys):
        log_step(step=1, action="test", reward=0.123456, done=False, error=None)
        out = capsys.readouterr().out.strip()
        assert "reward=0.12" in out

    def test_no_newlines_within_line(self, capsys):
        log_start(task="easy", env="dataqa_env", model="model")
        log_step(step=1, action="act", reward=0.0, done=False, error=None)
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        out = capsys.readouterr().out
        lines = [l for l in out.split("\n") if l.strip()]
        assert len(lines) == 3
        assert lines[0].startswith("[START]")
        assert lines[1].startswith("[STEP]")
        assert lines[2].startswith("[END]")
