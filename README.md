---
title: DataQA Environment Server
emoji: "\U0001F50D"
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

## Motivation

Every ML engineer and data scientist spends significant time debugging data quality issues — missing values, type mismatches, logical inconsistencies, and subtle statistical anomalies — before data enters ML pipelines or production databases. This is a genuine, high-frequency human task that directly impacts model quality and business outcomes.

DataQA turns this into a structured, gradable RL environment where agents must systematically inspect corrupted datasets, reason about schema constraints and validation rules, and pinpoint every planted issue — from obvious nulls to subtle data leakage signals that require domain expertise.

## Environment API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode with a corrupted dataset |
| `/step` | POST | Submit identified issues, receive scored feedback |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check |

## Tasks

| Task | Issues | Difficulty | Domain | Description |
|------|--------|-----------|--------|-------------|
| `easy` | 4 | Beginner | HR/Employee data | Nulls, wrong types, duplicates, out-of-range values |
| `medium` | 6 | Intermediate | E-commerce orders | Format violations, inconsistent computed fields, duplicate keys |
| `hard` | 8 | Advanced | ML experiment metadata | Data leakage signals, unreasonable GPU memory, timestamp ordering, whitespace-only fields |

**Difficulty progression**: Easy issues are individually obvious (empty fields, text in numeric columns). Medium issues require cross-column reasoning (total != qty * price) and set membership checks. Hard issues require ML domain knowledge (val_loss < train_loss = data leakage) and multi-row temporal reasoning.

## Action Space

The agent submits a list of issue strings, each in the format:
```
row:<row_number>,col:<column_name>,issue:<issue_type>
```

- `row_number`: 1-indexed position in the CSV data (after header). Row 1 = first data row.
- `column_name`: Exact column header name, lowercase.
- `issue_type`: One of the supported types below.

**Supported Issue Types:**

| Type | Description | Example |
|------|-------------|---------|
| `missing_value` | Null, empty, or whitespace-only | Empty name field |
| `wrong_type` | Value doesn't match expected type | Salary as "seventy-five thousand" |
| `duplicate_row` | Exact duplicate or duplicate key | Two rows with same employee_id |
| `out_of_range` | Value outside valid range | Salary of 5000 when min is 50000 |
| `format_violation` | Wrong format or invalid enum | Date as DD/MM/YYYY instead of YYYY-MM-DD |
| `inconsistent_value` | Computed field mismatch, logical inconsistency | total != qty * price |
| `statistical_outlier` | Unreasonable value given context | resnet18 using 42.5GB GPU |
| `referential_integrity` | Foreign key violation | (available for custom tasks) |

## Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `dataset_csv` | str | The corrupted dataset in CSV format |
| `schema_description` | str | Column types, ranges, and constraints |
| `validation_rules` | str | Business rules the data must satisfy |
| `task_description` | str | Task context and instructions |
| `feedback` | str | Results from previous step (TP/FP/FN counts, precision/recall) |
| `num_issues_hint` | int | Exact count of planted issues |
| `max_steps` | int | Maximum attempts allowed |
| `done` | bool | Whether episode has terminated |
| `reward` | float | Best weighted reward so far (0.0-1.0) |

**Observation Metadata** (available after each step):
- `f1`, `weighted_reward`, `precision`, `recall`
- `tp`, `fp`, `fn`
- `difficulty_found`, `difficulty_missed`

## Reward Function

### Difficulty-Weighted Reward (Primary)

Each planted issue has a **difficulty weight** (1.0-3.0) reflecting how hard it is to detect. The primary reward is a **weighted F1 score** that provides meaningful per-step partial progress signals:

| Weight | Category | Examples |
|--------|----------|----------|
| 1.0 | Easy | Missing values, obvious out-of-range, wrong type |
| 1.5-2.0 | Medium | Duplicate keys, format violations, cross-column checks |
| 2.5-3.0 | Hard | Data leakage, statistical outliers, whitespace-only |

**Formula:**
- **Weighted Recall** = (sum of difficulty weights for found issues) / (total difficulty weight)
- **Weighted Precision** = (found weight) / (found weight + FP count * avg difficulty)
- **Weighted F1** = harmonic mean of weighted precision and recall

This means:
- Finding a hard issue (difficulty 3.0) increases reward 3x more than finding an easy one (1.0)
- False positives are penalized proportionally to average issue difficulty
- The agent sees meaningful reward differences at every step, not just pass/fail

### Standard F1 (also computed)

Available in observation metadata for comparison. Uses unweighted set matching.

### Episode Boundaries

- Each task allows up to 3 steps (attempts)
- Episode ends when F1 >= 0.999 (perfect) or max steps reached
- Best score across all steps is the final reward (monotonically non-decreasing)
- Reward is always in [0.0, 1.0]

## Baseline Scores

Baseline scores using Qwen2.5-72B-Instruct via HuggingFace Router:

| Task | Expected Score Range | Description |
|------|---------------------|-------------|
| `easy` | 0.7 - 1.0 | Most LLMs find obvious issues reliably |
| `medium` | 0.5 - 0.8 | Cross-column reasoning is challenging |
| `hard` | 0.3 - 0.6 | ML domain knowledge and subtle patterns |

Scores vary by model capability. Frontier models (GPT-4, Claude) typically score higher on the hard task due to better domain reasoning.

## Extensibility

DataQA supports custom tasks, contamination rules, and difficulty levels via a programmatic API.

### Custom Contamination Rules

```python
from dataqa_env import register_contamination_rule
from dataqa_env.server.tasks import PlantedIssue

def swap_digits(rows, header, col_idx, row_idx, rng):
    val = rows[row_idx][col_idx]
    corrupted = val[::-1]
    issue = PlantedIssue(
        row=row_idx + 1, col=header[col_idx],
        issue_type="format_violation",
        description=f"Digits swapped in {header[col_idx]}",
        difficulty=2.0,
    )
    return corrupted, issue

register_contamination_rule("swap_digits", swap_digits)
```

### Custom Tasks from Config

```python
from dataqa_env import create_task_from_config, register_task

task = create_task_from_config(
    task_id="custom",
    name="Custom Validation",
    description="Find quality issues in this dataset.",
    schema_description="id: int, name: str, score: int (0-100)",
    validation_rules="No missing values. Scores must be 0-100.",
    clean_csv="id,name,score\n1,Alice,95\n2,Bob,87\n3,Carol,92",
    contaminations=[
        {"rule": "missing_value", "row": 0, "col": 1, "difficulty": 1.0},
        {"rule": "negative_value", "row": 2, "col": 2, "difficulty": 1.5},
    ],
)
register_task("custom", lambda seed: task)
```

### Built-in Contamination Rules

| Rule | Effect | Default Difficulty |
|------|--------|--------------------|
| `missing_value` | Sets field to empty string | 1.0 |
| `whitespace_value` | Sets field to single space | 2.5 |
| `wrong_type_text` | Replaces with random text ("N/A", "null", etc.) | 1.0 |
| `negative_value` | Negates numeric value | 1.0 |

## Quick Start

```bash
# Install
pip install -e .

# Run server locally
uvicorn dataqa_env.server.app:app --host 0.0.0.0 --port 8000

# Run inference (set your API credentials)
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=your-token \
python inference.py
```

## Docker

```bash
docker build -t dataqa-env .
docker run -p 8000:8000 dataqa-env
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

89 tests covering:
- Task creation, corruption, and issue planting (difficulty weights, seed determinism)
- Issue key parsing (standard, lenient, edge cases)
- F1 and difficulty-weighted reward computation
- Full environment reset/step lifecycle
- Inference script parsing and prompt building
- **Structured log format** ([START], [STEP], [END] — exact field names and ordering)
- Score bounds (0.0-1.0), best-score monotonicity
- Extensibility API (custom rules, custom tasks, environment integration)

## Validation

```bash
# OpenEnv spec validation
openenv validate .

# Pre-submission validation (requires HF Space URL)
./prevalidation_script.sh https://your-space.hf.space
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace token / API key | - |
| `ENV_URL` | Environment server URL | `http://localhost:8000` |

## Architecture

```
dataqa_env/
├── __init__.py            # Public API + extensibility exports
├── models.py              # Pydantic: DataQAAction, DataQAObservation, DataQAState
├── client.py              # EnvClient for WebSocket connections
├── server/
│   ├── environment.py     # Core DataQAEnvironment (reset/step/state + weighted rewards)
│   ├── tasks.py           # Task definitions + contamination rules + extensibility API
│   ├── app.py             # FastAPI server (via openenv-core create_app)
│   └── Dockerfile
tests/
├── test_tasks.py          # Task creation, corruption, difficulty weights
├── test_environment.py    # Environment lifecycle, scoring, metadata
├── test_inference.py      # LLM response parsing, prompt building, log format
└── test_extensibility.py  # Custom rules, custom tasks, registration API
inference.py               # Baseline LLM agent (OpenAI client, structured logs)
openenv.yaml               # OpenEnv/HF Spaces spec
pyproject.toml             # Package metadata and dependencies
Dockerfile                 # Production container
```
