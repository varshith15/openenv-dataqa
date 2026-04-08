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

## Environment API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode with `{"task_id": "easy"}` |
| `/step` | POST | Submit identified issues + proposed fixes |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check |

## Action Format

**Identify:** `row:<N>,col:<column>,issue:<type>` where type is one of: `missing_value`, `wrong_type`, `duplicate_row`, `out_of_range`, `format_violation`, `inconsistent_value`, `statistical_outlier`

**Fix:** `row:<N>,col:<column>,fix:<corrected_value>`

Both can be submitted in the same step or across multiple steps (3 steps max).

## Reward Design

| Property | Detail |
|----------|--------|
| Range | Strict (0, 1) — 0.001 minimum, 0.999 maximum |
| Partial credit | Yes — per-issue, difficulty-weighted |
| Monotonic | Best score across all steps is final reward |
| Penalizes guessing | False positives reduce precision, fixing non-issues scores 0 |
| Multi-step improvement | Detailed feedback enables learning across attempts |

**Fix grading tiers** (by issue type):
- Exact match with clean value → 1.0
- Valid fix: right type/range, addresses the issue → 0.8
- Partially valid: reasonable attempt, right direction → 0.4
- Right cell, wrong value → 0.1
- Non-issue cell → 0.0

## Quick Start

```bash
pip install -e .
uvicorn dataqa_env.server.app:app --host 0.0.0.0 --port 8000

# Run baseline agent
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=your-token \
python inference.py
```

## Testing

128 tests covering task creation, reward computation, fix grading, environment lifecycle, inference parsing, and extensibility API.

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Architecture

```
dataqa_env/
├── models.py              # DataQAAction (issues + fixes), DataQAObservation
├── server/
│   ├── environment.py     # Two-phase grading engine (identify + fix + combined reward)
│   ├── tasks.py           # 7 task definitions + contamination rules + extensibility API
│   └── app.py             # FastAPI server (via openenv-core)
inference.py               # Two-phase baseline agent (identify → fix)
tests/                     # 128 tests
```
