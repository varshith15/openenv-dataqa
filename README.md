---
title: DataQA Environment Server
emoji: "\U0001F50D"
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# DataQA Environment

A two-phase OpenEnv RL environment for **Data Quality Assurance** — an LLM agent inspects corrupted datasets, identifies all planted quality issues, and proposes data repairs.

### Demo: Agent Trajectory Replay

**Easy task** — Agent finds all 4 issues and proposes fixes (step 2):

![Easy task: all issues found + fixes proposed](docs/demo_easy.png)

**Hard task** — Agent identifies 8 subtle ML issues including data leakage and GPU memory outlier, proposes fixes (step 2):

![Hard task: ML experiment metadata with 8 issues](docs/demo_hard.png)

Green cells = correctly found issues. Yellow = missed. Green outlines = correct fixes with proposed values shown inline (e.g. `empty → David Kim`, `seventy-five thousand → 75000`).

> The interactive replay UI is available at the `/web` endpoint on the HF Space.

## Motivation

Every ML engineer and data scientist spends significant time debugging data quality issues — missing values, type mismatches, logical inconsistencies, and subtle statistical anomalies — before data enters ML pipelines or production databases. This is a genuine, high-frequency human task that directly impacts model quality and business outcomes.

DataQA turns this into a **two-phase RL challenge**:
1. **Identify** — systematically inspect corrupted data and pinpoint every planted issue
2. **Fix** — propose corrected values by reasoning about schema, constraints, and context

This creates a rich multi-step decision problem where agents must explore datasets strategically, distinguish subtle anomalies from noise, and reason about what the correct data should be.

## Environment API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode with a corrupted dataset |
| `/step` | POST | Submit identified issues + proposed fixes |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check |

## Tasks

| Task | Issues | Difficulty | Domain | Description |
|------|--------|-----------|--------|-------------|
| `easy` | 6 | Beginner | HR/Employee data (21 rows) | Nulls, wrong types, duplicates, out-of-range, email-name mismatch, future dates |
| `medium` | 8 | Intermediate | E-commerce orders (31 rows) | Inconsistent totals, invalid categories, duplicate keys, wrong date formats, invalid country codes, future-date deliveries |
| `hard` | 10 | Advanced | ML experiment metadata (31 rows) | Data leakage signals, unreasonable GPU memory, impossibly fast training, SOTA-exceeding accuracy, timestamp ordering, whitespace-only fields |

**Difficulty progression**: Easy issues are individually obvious (empty fields, text in numeric columns). Medium issues require cross-column reasoning (total != qty * price) and set membership checks. Hard issues require ML domain knowledge (val_loss < train_loss = data leakage) and multi-row temporal reasoning.

## Two-Phase Action Space

### Phase 1: Identify Issues

Submit issues in format: `row:<row_number>,col:<column_name>,issue:<issue_type>`

- `row_number`: 1-indexed data row position (after header)
- `column_name`: Exact column header name, lowercase
- `issue_type`: One of the supported types below

### Phase 2: Propose Fixes

Submit fixes in format: `row:<row_number>,col:<column_name>,fix:<corrected_value>`

The agent proposes the **correct value** that should replace the corrupted data. Fixes are graded against the original clean dataset.

Both phases can be submitted in the same step or across multiple steps.

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

| Field | Type | Description |
|-------|------|-------------|
| `dataset_csv` | str | The corrupted dataset in CSV format |
| `schema_description` | str | Column types, ranges, and constraints |
| `validation_rules` | str | Business rules the data must satisfy |
| `task_description` | str | Task context and instructions |
| `feedback` | str | Per-step results: TP/FP/FN, precision/recall, fix scores |
| `num_issues_hint` | int | Exact count of planted issues |
| `max_steps` | int | Maximum attempts allowed |
| `done` | bool | Whether episode has terminated |
| `reward` | float | Best combined reward so far (0.0-1.0) |

**Observation Metadata** (per step):
- Identify: `identify_f1`, `identify_score`, `precision`, `recall`, `tp`, `fp`, `fn`
- Fix: `fix_score`, `fixes_correct`, `fixes_partial`, `fixes_wrong`, `fixes_attempted`
- Combined: `combined_reward`, `difficulty_found`, `difficulty_missed`

## Reward Function

### Combined Reward

```
combined_reward = 0.6 * identify_score + 0.4 * fix_score
```

If no fixes are submitted, `combined_reward = identify_score` (no penalty — backward compatible).

### Identify Score (Difficulty-Weighted F1)

Each planted issue has a **difficulty weight** (1.0-3.0):

| Weight | Category | Examples |
|--------|----------|----------|
| 1.0 | Easy | Missing values, obvious out-of-range, wrong type |
| 1.5-2.0 | Medium | Duplicate keys, format violations, cross-column checks |
| 2.5-3.0 | Hard | Data leakage, statistical outliers, whitespace-only |

- **Weighted Recall** = (difficulty of found issues) / (total difficulty)
- **Weighted Precision** = penalizes false positives proportional to average difficulty
- **Weighted F1** = harmonic mean

### Fix Score (Difficulty-Weighted Quality)

Each proposed fix is compared against the original clean value:

| Fix Quality | Score | Description |
|-------------|-------|-------------|
| Exact match | 1.0 | Case-insensitive, whitespace-stripped match |
| Numeric close | 0.8 | Within 1% of correct numeric value |
| Correct cell | 0.1 | Right location, wrong value |
| Non-issue cell | 0.0 | Fix targets a cell with no issue |

Fix score = (sum of best fix score per issue × difficulty weight) / (total difficulty weight)

### Reward Properties

- **Per-step partial progress**: reward increases as more issues are found/fixed
- **Difficulty-aware**: finding subtle issues earns more than obvious ones
- **Penalizes bad behavior**: false positives reduce score, fixing non-issues earns nothing
- **Monotonically non-decreasing**: best score across all steps is the final reward
- **Always in [0.0, 1.0]**: meets hackathon requirement

### Episode Boundaries

- Each task allows up to 3 steps (attempts)
- Episode ends when F1 >= 0.999 (perfect identification) or max steps reached
- Agent receives detailed feedback after each step to improve on next attempt

## Baseline Scores

Baseline agent uses Qwen2.5-72B-Instruct via HuggingFace Router:

| Task | Identify Score | Fix Score | Combined | Notes |
|------|---------------|-----------|----------|-------|
| `easy` | 0.7-1.0 | 0.5-0.9 | 0.6-1.0 | Most LLMs find obvious issues reliably |
| `medium` | 0.5-0.8 | 0.3-0.6 | 0.4-0.7 | Cross-column reasoning challenges models |
| `hard` | 0.3-0.6 | 0.2-0.4 | 0.3-0.5 | ML domain knowledge and subtle patterns |

Scores vary by model. The hard task is designed to challenge frontier models.

## Extensibility

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
| `wrong_type_text` | Replaces with random text | 1.0 |
| `negative_value` | Negates numeric value | 1.0 |

## Setup & Quick Start

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

118 tests covering:
- Task creation, corruption, and difficulty weights
- Issue key and fix parsing (standard, lenient, edge cases)
- F1, weighted reward, and fix quality computation
- Full environment lifecycle (identify-only and identify+fix)
- Combined reward calculation and weight verification
- Inference script parsing and prompt building
- Structured log format ([START], [STEP], [END])
- Score bounds (0.0-1.0), best-score monotonicity
- Extensibility API (custom rules, custom tasks)

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
├── models.py              # Pydantic: DataQAAction (issues + fixes), DataQAObservation, DataQAState
├── client.py              # EnvClient for WebSocket connections
├── server/
│   ├── environment.py     # Two-phase DataQAEnvironment (identify + fix + combined reward)
│   ├── tasks.py           # Task definitions + contamination rules + extensibility API
│   ├── app.py             # FastAPI server (via openenv-core create_app)
│   └── Dockerfile
tests/
├── test_tasks.py          # Task creation, corruption, difficulty weights
├── test_environment.py    # Identify scoring, fix grading, combined reward, lifecycle
├── test_inference.py      # LLM response parsing, fix parsing, prompt building, log format
└── test_extensibility.py  # Custom rules, custom tasks, registration API
inference.py               # Two-phase baseline agent (identify → fix)
openenv.yaml               # OpenEnv/HF Spaces spec
pyproject.toml             # Package metadata and dependencies
Dockerfile                 # Production container
```
