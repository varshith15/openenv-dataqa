"""
DataQAEnv Client
----------------
Client-side wrapper for the DataQA environment server.
"""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import DataQAAction, DataQAObservation, DataQAState


class DataQAEnv(EnvClient[DataQAAction, DataQAObservation, DataQAState]):

    def _step_payload(self, action: DataQAAction) -> dict:
        return {"issues": action.issues, "task_id": action.task_id}

    def _parse_result(self, payload: dict) -> StepResult[DataQAObservation]:
        obs = DataQAObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> DataQAState:
        return DataQAState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            current_step=payload.get("current_step", 0),
            max_steps=payload.get("max_steps", 3),
            best_score=payload.get("best_score", 0.0),
            total_planted_issues=payload.get("total_planted_issues", 0),
        )
