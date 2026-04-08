---
title: DataQA Environment Server
emoji: "\U0001F50D"
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# DataQA Environment

**A two-phase OpenEnv RL environment for Data Quality Assurance** — an LLM agent inspects corrupted datasets, identifies all planted quality issues, and proposes data repairs.

## Why DataQA? The Moat

### 1. Solves a Real, High-Frequency Problem

Every ML team burns hours on data quality — missing values, type mismatches, logical inconsistencies, subtle statistical anomalies — before data enters training pipelines or production databases. DataQA turns this universal pain point into a graded RL environment. Unlike synthetic toy problems, **these are the exact data bugs that corrupt production ML models.**

### 2. Seven Diverse Domains, One Unified Interface

| Task | Domain | Issues | What Makes It Hard |
|------|--------|--------|--------------------|
| `easy` | HR / Employee data | 6 | Missing values, typos, format errors |
| `medium` | E-commerce orders | 8 | Cross-column math (`total != qty * price`), OCR errors |
| `hard` | ML experiment metadata | 10 | Data leakage detection, impossible GPU specs, SOTA violations |
| `alignment` | LLM fine-tuning data (NVIDIA HelpSteer) | 12 | Hallucinated citations, self-contradictions, toxic content scored as helpful |
| `coding` | Code instruction-response pairs | 10 | Logic bugs in "correct" code, `eval()` injection, language mismatches |
| `toolcalling` | Function-calling schemas | 10 | Hallucinated parameters, missing required args, name mismatches |
| `moderation` | Content moderation labels | 10 | Mislabeled hate speech, false positives on clean text |

**66 total planted issues** spanning tabular data, free-text, code, JSON schemas, and safety labels. No other OpenEnv submission covers this breadth with a single coherent reward function.

### 3. Two-Phase Reward — Identify Then Fix

Most data QA environments only ask "is there a bug?" DataQA goes further:

- **Phase 1 (Identify):** Find all issues — graded by difficulty-weighted F1
- **Phase 2 (Fix):** Propose the correct value — graded against the clean original with tiered scoring (exact match = 1.0, valid fix = 0.8, partial = 0.4, right cell wrong value = 0.1)

```
combined_reward = 0.6 * identify_score + 0.4 * fix_score
```

This creates a richer learning signal than binary classification. An agent that finds 8/10 issues and fixes 5 of them correctly gets meaningful partial credit — perfect for GRPO/RLHF training.

### 4. Difficulty-Weighted Scoring Rewards Deeper Reasoning

Each planted issue has a difficulty weight (1.0-3.0). Finding a hallucinated citation (3.0) earns triple the reward of finding an empty field (1.0). This incentivizes agents to develop genuine reasoning capabilities rather than pattern-matching surface-level errors.

### 5. Multi-Step Feedback Loop

Agents get 3 attempts per task with detailed per-step feedback:
- Which issues were correct (true positives) vs wrong (false positives)
- Which issues were missed (false negatives) with difficulty hints
- Fix quality scores with reasons

This enables the agent to **learn from its mistakes within a single episode** — a natural curriculum.

### 6. Fully Extensible

```python
# Add your own contamination rules
register_contamination_rule("swap_digits", my_swap_fn)

# Create tasks from any CSV
task = create_task_from_config(
    task_id="custom", clean_csv="...",
    contaminations=[{"rule": "missing_value", "row": 0, "col": 1}]
)
register_task("custom", lambda seed: task)
```

New domains can be added in minutes. The contamination engine is domain-agnostic.

---

## Demo: Agent Trajectory

```
HARD TASK — ML experiment metadata
  Step 1: Found 5/10, missed hard issues    → Reward: 0.69
  Step 2: Found 10/10 + 5 fixes proposed   → Reward: 0.77
  Issues requiring ML knowledge:
    • val_loss < train_loss (data leakage signal)
    • resnet18 using 42.5GB GPU (impossible for 11M params)
    • 350 epochs on ImageNet in 30 min (impossibly fast)
    • wav2vec2 at 98.5% accuracy (exceeds SOTA)

ALIGNMENT TASK — NVIDIA HelpSteer data
  Step 1: Found 7/12, missed subtle issues  → Reward: 0.58
  Step 2: Found 12/12 + 3 fixes proposed   → Reward: 0.72
  Issues requiring deep reasoning:
    • Cerasus vs Prunus serrulata (wrong taxonomic name)
    • $400.3M at Sotheby's vs $450.3M at Christie's (close but wrong)
    • Fake Nature paper by "Dr. Sarah Chen" (hallucinated citation)
    • Gender-biased advice rated helpfulness=4 (toxic content with inflated scores)

CODING TASK — Code instruction-response pairs
  Issues requiring code understanding:
    • Binary search off-by-one (lo=mid causes infinite loop) marked correct
    • eval(uid) in Flask route — code injection vulnerability
    • JavaScript response for a Python-labeled task
    • Duplicate "merge sort" instruction across rows
```

> The interactive replay UI with color-coded dataset visualization is available on the HF Space.

## Environment API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode with a corrupted dataset |
| `/step` | POST | Submit identified issues + proposed fixes |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check |

## Tasks

**Difficulty progression**: Easy issues are individually obvious (empty fields, text in numeric columns). Medium issues require cross-column reasoning (total != qty * price) and set membership checks. Hard issues require ML domain knowledge (val_loss < train_loss = data leakage). Expert tasks (alignment, coding, toolcalling, moderation) require domain expertise, semantic reasoning, and cross-row comparison.

### Alignment Task: LLM Training Data Quality (Expert)

Built on **real data from [NVIDIA HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer)** — 30 human-annotated prompt-response pairs with quality scores (helpfulness, correctness, coherence, complexity, verbosity on 0-4 scale).

This task targets a critical real-world problem: **catching quality issues in LLM fine-tuning data before it corrupts model training**. The 12 planted issues represent failure modes actually seen in production data pipelines:

| Issue | Difficulty | Why It's Hard |
|---|---|---|
| Subtle factual error (*Cerasus* vs *Prunus serrulata*) | 3.0 | Old taxonomic synonym — sounds plausible, requires domain knowledge |
| Plausible wrong numbers ($400.3M at Sotheby's vs $450.3M at Christie's) | 3.0 | Right painting, wrong price by $50M and wrong auction house |
| Self-contradictory reasoning ("does NOT learn via backprop" then describes backprop) | 3.0 | Response negates its own conclusion — trains confused models |
| Hallucinated citation (fake Nature paper by fake Dr. Sarah Chen) | 3.0 | Fabricated study with specific fake statistics — most dangerous for training |
| Harmful coding advice ("use bare except everywhere") with high quality scores | 3.0 | Teaches dangerous practices if used for fine-tuning |
| Toxic/biased response scored as helpful | 3.0 | Gender-biased stereotypes with helpfulness=4 — poisons alignment training |
| Leaked system prompt (`[SYSTEM] You are a helpful AI...`) in response | 2.5 | Data pipeline failed to strip prompt template |
| Semantic near-duplicate prompt (rephrased, not exact copy) | 2.5 | Requires semantic similarity detection, not just string matching |
| Truncated response (cut mid-sentence) | 2.5 | `max_length` truncation without sentence boundary detection |
| Response in French for English prompt | 2.0 | Language contamination from multilingual training data |
| Response plagiarized from another row | 2.0 | Data pipeline shuffling/dedup failure |
| Whitespace-only prompt | 2.0 | Empty training example from pipeline artifact |

### Coding Task: Code Quality (Expert)

20-row dataset of code instruction-response pairs (Python algorithms, data structures, web, design patterns). 10 planted issues:

- Syntax errors in "correct" code (unbalanced parens)
- Logic bugs marked `is_correct=true` (binary search off-by-one infinite loop)
- Security vulnerabilities (`eval()` on user input) marked correct
- Language mismatches (JavaScript response labeled Python)
- Truncated code, difficulty label mismatches, duplicate instructions, wrong categories, missing test cases

### Tool-Calling Task: Function Schema Quality (Expert)

20-row dataset of function definitions with parameter schemas, example calls, and outputs. 10 planted issues:

- Function name mismatch between definition and example call
- Missing required parameters in example call
- Hallucinated parameters not in schema
- Type mismatches (string "high" for integer quality parameter)
- Invalid JSON, duplicate function names, misleading descriptions, wrong categories

### Moderation Task: Content Label Quality (Expert)

30-row dataset modeled on content moderation pipelines. 10 planted issues:

- Mislabeled hate speech and violence (unflagged toxic content)
- False positives on clean text (idioms flagged as hate)
- Subset rule violations (`hate_threatening` without `hate` flag)
- Out-of-range label values

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
| `reward` | float | Best combined reward so far (strict 0-1 range) |

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
| 2.5-3.0 | Hard | Data leakage, statistical outliers, hallucinated citations |

- **Weighted Recall** = (difficulty of found issues) / (total difficulty)
- **Weighted Precision** = penalizes false positives proportional to average difficulty
- **Weighted F1** = harmonic mean

### Fix Score (Tiered Grading by Issue Type)

Each proposed fix is graded with tiered scoring that gives partial credit for reasonable attempts:

| Fix Quality | Score | Description |
|-------------|-------|-------------|
| Exact match | 1.0 | Case-insensitive, whitespace-stripped match with clean value |
| Valid fix | 0.8 | Right type/range, addresses the issue (e.g., any non-empty value for missing field) |
| Partially valid | 0.4 | Reasonable attempt, right direction (e.g., numeric in right ballpark) |
| Right cell, wrong value | 0.1 | Targets correct cell but fix doesn't address the issue |
| Non-issue cell | 0.0 | Fix targets a cell with no issue |

Fix score = (sum of best fix score per issue x difficulty weight) / (total difficulty weight)

### Reward Properties

| Property | Detail |
|----------|--------|
| Range | Strict (0, 1) — 0.001 minimum, 0.999 maximum |
| Partial credit | Yes — per-issue, difficulty-weighted |
| Monotonic | Best score across all steps is final reward |
| Penalizes guessing | False positives reduce precision, fixing non-issues scores 0 |
| Multi-step improvement | Detailed feedback enables learning across attempts |

### Episode Boundaries

- Each task allows up to 3 steps (attempts)
- Episode ends when F1 >= 0.999 (perfect identification) or max steps reached
- Agent receives detailed feedback after each step to improve on next attempt

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

128 tests covering:
- Task creation, corruption, and difficulty weights for all 7 tasks
- Issue key and fix parsing (standard, lenient, edge cases)
- F1, weighted reward, and fix quality computation
- Full environment lifecycle (identify-only and identify+fix)
- Combined reward calculation and weight verification
- Inference script parsing and prompt building
- Structured log format ([START], [STEP], [END])
- Score bounds (strict 0-1), best-score monotonicity
- Extensibility API (custom rules, custom tasks)
- Moderation task determinism and label consistency

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
│   ├── tasks.py           # 7 task definitions + contamination rules + extensibility API
│   ├── gradio_ui.py       # Interactive web UI with agent trajectory replay
│   ├── app.py             # FastAPI server (via openenv-core create_app)
│   └── Dockerfile
tests/
├── test_tasks.py          # Task creation, corruption, difficulty weights (all 7 tasks)
├── test_environment.py    # Identify scoring, fix grading, combined reward, lifecycle
├── test_inference.py      # LLM response parsing, fix parsing, prompt building, log format
└── test_extensibility.py  # Custom rules, custom tasks, registration API
inference.py               # Two-phase baseline agent (identify then fix)
openenv.yaml               # OpenEnv/HF Spaces spec
pyproject.toml             # Package metadata and dependencies
Dockerfile                 # Production container
```

### Key Modules

**`dataqa_env/server/tasks.py`** — The core of the environment. Each task function (`create_task_easy`, `create_task_coding`, etc.) builds a clean CSV dataset, injects corruptions as `PlantedIssue` objects with row/col/type/difficulty, and returns a `Task` dataclass. The `TASK_REGISTRY` dict maps task IDs to factory functions. The extensibility API (`register_task`, `register_contamination_rule`, `create_task_from_config`) allows users to add domains without modifying source.

**`dataqa_env/server/environment.py`** — The `DataQAEnvironment` class inherits from OpenEnv's `Environment` base. `reset()` loads a task by ID and returns the corrupted CSV + schema. `step()` parses issue keys and fix proposals from the action, computes difficulty-weighted F1 for identification, grades fixes with tiered scoring by issue type, and returns combined reward with detailed feedback. Handles HTTP statelessness via auto-reset from `action.task_id`.

**`dataqa_env/models.py`** — Pydantic models for the OpenEnv interface. `DataQAAction` carries `issues: List[str]`, `fixes: List[str]`, and `task_id: str`. `DataQAObservation` carries the CSV, schema, rules, feedback, and scoring metadata. `DataQAState` tracks episode progress.

**`inference.py`** — Baseline LLM agent using OpenAI-compatible API. Runs all 7 tasks sequentially with 3 steps each. Lenient regex parsing handles case variations and delimiter differences in LLM output. Structured logging in `[START]/[STEP]/[END]` format for evaluation.
