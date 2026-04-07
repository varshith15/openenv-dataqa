---
title: DataQA Environment Server
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DataQA Environment

An OpenEnv environment for **Data Quality Assurance** — an LLM agent inspects datasets with planted quality issues and must identify them all.

## Overview

DataQA simulates the real-world task of validating datasets before they enter ML training pipelines or production databases. The agent receives a corrupted dataset along with its schema and validation rules, then must identify all planted data quality issues.

### Why Data QA?

Every ML engineer and data scientist spends significant time debugging data quality issues — missing values, type mismatches, inconsistencies, and subtle statistical anomalies. This environment turns that task into a structured, gradable challenge.

## Environment API

| Endpoint | Description |
|----------|-------------|
| `reset(task_id)` | Start a new episode with a corrupted dataset |
| `step(issues)` | Submit identified issues, receive F1-scored feedback |
| `state()` | Get current episode state |

## Tasks

| Task | Issues | Difficulty | Description |
|------|--------|-----------|-------------|
| `easy` | 4 | Beginner | Employee directory — nulls, wrong types, duplicates, out-of-range |
| `medium` | 6 | Intermediate | E-commerce orders — format violations, inconsistent totals, duplicate keys |
| `hard` | 8 | Advanced | ML experiment metadata — data leakage signals, unreasonable GPU usage, timestamp ordering |

## Reward Function

Scoring uses **F1 score** (harmonic mean of precision and recall):

- **Precision**: What fraction of reported issues are real?
- **Recall**: What fraction of planted issues did the agent find?
- **F1**: `2 * precision * recall / (precision + recall)`

Issues are matched by `row:<N>,col:<column>,issue:<type>` keys.

The agent gets up to 3 attempts per task with feedback on each attempt (true positives, false positives, missed count).

## Action/Observation Space

**Action**: List of issue strings in format `row:<row_number>,col:<column_name>,issue:<issue_type>`

**Observation**: Dataset CSV + schema + validation rules + feedback from previous attempt

**Issue Types**: `missing_value`, `wrong_type`, `duplicate_row`, `out_of_range`, `format_violation`, `inconsistent_value`, `statistical_outlier`, `referential_integrity`

## Quick Start

```bash
# Install
pip install -e .

# Run server locally
uvicorn dataqa_env.server.app:app --host 0.0.0.0 --port 8000

# Run inference
API_BASE_URL=https://api.groq.com/openai/v1 \
MODEL_NAME=llama-3.3-70b-versatile \
LLM_API_KEY=your-key \
python inference.py
```

## Docker

```bash
docker build -t dataqa-env -f dataqa_env/server/Dockerfile .
docker run -p 8000:8000 dataqa-env
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | Model identifier | `llama-3.3-70b-versatile` |
| `HF_TOKEN` | HuggingFace token | - |
| `ENV_URL` | Environment server URL | `http://localhost:8000` |
| `LLM_API_KEY` | API key for LLM provider | Falls back to HF_TOKEN |

## Architecture

```
dataqa_env/
├── models.py              # Pydantic: DataQAAction, DataQAObservation, DataQAState
├── client.py              # EnvClient for WebSocket connections
├── server/
│   ├── environment.py     # Core DataQAEnvironment (reset/step/state)
│   ├── tasks.py           # Task definitions + data corruption + grading
│   ├── app.py             # FastAPI server
│   └── Dockerfile
├── openenv.yaml
├── pyproject.toml
└── inference.py           # LLM agent using OpenAI client
```
