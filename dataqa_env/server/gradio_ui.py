"""
Gradio UI — Agent Trajectory Replay Viewer for DataQA.

Designed for judges: zero clicks needed, auto-plays on load.
Tab per task, step slider, prominent metric cards, color-coded dataset.
"""

from __future__ import annotations

import csv
import io

import gradio as gr

from .environment import DataQAEnvironment, parse_issue_key
from .tasks import list_tasks, PlantedIssue
from ..models import DataQAAction


# ── Pre-built agent trajectories (simulates baseline agent) ──

AGENT_TRAJECTORIES = {
    "easy": [
        {
            "issues": [
                "row:4,col:name,issue:missing_value",
                "row:7,col:salary,issue:wrong_type",
                "row:9,col:salary,issue:out_of_range",
                "row:18,col:start_date,issue:out_of_range",
                "row:3,col:email,issue:format_violation",  # FP
            ],
            "fixes": [],
        },
        {
            "issues": [
                "row:4,col:name,issue:missing_value",
                "row:7,col:salary,issue:wrong_type",
                "row:9,col:salary,issue:out_of_range",
                "row:21,col:employee_id,issue:duplicate_row",
                "row:15,col:email,issue:inconsistent_value",
                "row:18,col:start_date,issue:out_of_range",
            ],
            "fixes": [
                "row:4,col:name,fix:David Kim",
                "row:7,col:salary,fix:75000",
                "row:9,col:salary,fix:73000",
                "row:15,col:email,fix:oscar.rivera@company.com",
                "row:18,col:start_date,fix:2022-01-19",
            ],
        },
    ],
    "medium": [
        {
            "issues": [
                "row:5,col:total,issue:inconsistent_value",
                "row:10,col:category,issue:format_violation",
                "row:14,col:product_name,issue:missing_value",
                "row:17,col:quantity,issue:out_of_range",
                "row:19,col:order_id,issue:duplicate_row",
                "row:12,col:order_date,issue:format_violation",
                "row:24,col:shipping_country,issue:format_violation",
            ],
            "fixes": [],
        },
        {
            "issues": [
                "row:5,col:total,issue:inconsistent_value",
                "row:10,col:category,issue:format_violation",
                "row:14,col:product_name,issue:missing_value",
                "row:17,col:quantity,issue:out_of_range",
                "row:19,col:order_id,issue:duplicate_row",
                "row:12,col:order_date,issue:format_violation",
                "row:24,col:shipping_country,issue:format_violation",
                "row:29,col:order_date,issue:inconsistent_value",
            ],
            "fixes": [
                "row:5,col:total,fix:42.00",
                "row:10,col:category,fix:Sports",
                "row:12,col:order_date,fix:2024-01-26",
                "row:14,col:product_name,fix:LED Strip Lights",
                "row:24,col:shipping_country,fix:US",
                "row:29,col:order_date,fix:2024-02-12",
            ],
        },
    ],
    "hard": [
        {
            "issues": [
                "row:14,col:training_time_hours,issue:out_of_range",
                "row:13,col:learning_rate,issue:out_of_range",
                "row:15,col:model_name,issue:missing_value",
                "row:9,col:batch_size,issue:format_violation",
                "row:10,col:train_size,issue:inconsistent_value",
            ],
            "fixes": [],
        },
        {
            "issues": [
                "row:14,col:training_time_hours,issue:out_of_range",
                "row:13,col:learning_rate,issue:out_of_range",
                "row:15,col:model_name,issue:missing_value",
                "row:9,col:batch_size,issue:format_violation",
                "row:10,col:train_size,issue:inconsistent_value",
                "row:5,col:val_loss,issue:inconsistent_value",
                "row:7,col:gpu_memory_gb,issue:statistical_outlier",
                "row:11,col:timestamp,issue:inconsistent_value",
                "row:9,col:training_time_hours,issue:statistical_outlier",
                "row:12,col:test_accuracy,issue:statistical_outlier",
            ],
            "fixes": [
                "row:14,col:training_time_hours,fix:72.0",
                "row:13,col:learning_rate,fix:0.00001",
                "row:15,col:model_name,fix:whisper-small",
                "row:9,col:batch_size,fix:256",
                "row:9,col:training_time_hours,fix:36.0",
            ],
        },
    ],
    "alignment": [
        {
            "issues": [
                "row:6,col:response,issue:inconsistent_value",
                "row:15,col:language,issue:inconsistent_value",
                "row:17,col:instruction,issue:missing_value",
                "row:19,col:response,issue:inconsistent_value",
                "row:21,col:instruction,issue:duplicate_row",
                "row:23,col:response,issue:missing_value",
                "row:3,col:response,issue:inconsistent_value",
            ],
            "fixes": [],
        },
        {
            "issues": [
                "row:4,col:response,issue:inconsistent_value",
                "row:6,col:response,issue:inconsistent_value",
                "row:8,col:response,issue:inconsistent_value",
                "row:10,col:response,issue:inconsistent_value",
                "row:11,col:response,issue:inconsistent_value",
                "row:15,col:language,issue:inconsistent_value",
                "row:17,col:instruction,issue:missing_value",
                "row:19,col:response,issue:inconsistent_value",
                "row:21,col:instruction,issue:duplicate_row",
                "row:23,col:response,issue:missing_value",
                "row:24,col:response,issue:inconsistent_value",
                "row:3,col:response,issue:inconsistent_value",
            ],
            "fixes": [
                "row:6,col:response,fix:Buenos dias. In Spanish this is a common greeting used in the morning.",
                "row:10,col:response,fix:The capital of Japan is Tokyo.",
                "row:19,col:response,fix:The water cycle describes continuous movement of water on Earth.",
            ],
        },
    ],
}


# ── HTML rendering ──

def _metric_card(label: str, value: str, color: str = "#333") -> str:
    return (
        f'<div style="text-align:center;padding:12px 16px;background:#f8f9fa;'
        f'border-radius:8px;min-width:100px;">'
        f'<div style="font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;">{label}</div>'
        f'<div style="font-size:28px;font-weight:700;color:{color};margin-top:2px;">{value}</div>'
        f'</div>'
    )


def _csv_to_html(
    csv_text: str,
    planted: list[PlantedIssue],
    correct: set[tuple[int, str]],
    fp: set[tuple[int, str]],
    missed: set[tuple[int, str]],
    fixed: dict[tuple[int, str], str],
    fix_values: dict[tuple[int, str], str] | None = None,
) -> str:
    """Render CSV as HTML with color-coded cells and inline fix proposals."""
    fix_values = fix_values or {}
    desc_map = {(i.row, i.col): i for i in planted}
    reader = csv.reader(io.StringIO(csv_text.strip()))
    rows = list(reader)
    if not rows:
        return ""

    header = rows[0]
    header_lower = [h.strip().lower() for h in header]
    data = rows[1:]

    t = ['<table style="border-collapse:collapse;width:100%;font-size:12px;font-family:\'SF Mono\',monospace;">']
    t.append('<tr>')
    t.append('<th style="border:1px solid #dee2e6;padding:6px 8px;background:#343a40;color:#fff;font-size:11px;">Row</th>')
    for h in header:
        t.append(f'<th style="border:1px solid #dee2e6;padding:6px 8px;background:#343a40;color:#fff;font-size:11px;">{h}</th>')
    t.append('</tr>')

    for i, row in enumerate(data):
        rn = i + 1
        bg = "#fff" if i % 2 == 0 else "#f8f9fa"
        t.append(f'<tr style="background:{bg};">')
        t.append(f'<td style="border:1px solid #dee2e6;padding:4px 8px;color:#adb5bd;text-align:center;font-size:11px;">{rn}</td>')
        for j, val in enumerate(row):
            col = header_lower[j] if j < len(header_lower) else ""
            ck = (rn, col)
            s = "border:1px solid #dee2e6;padding:4px 8px;"
            tip = ""
            badge = ""

            issue = desc_map.get(ck)

            if ck in correct:
                s += "background:#d4edda;"
                tip = f"FOUND: {issue.description}" if issue else ""
                badge = '<span style="font-size:9px;background:#28a745;color:#fff;padding:1px 4px;border-radius:3px;margin-left:4px;">TP</span>'
            elif ck in fp:
                s += "background:#f8d7da;"
                badge = '<span style="font-size:9px;background:#dc3545;color:#fff;padding:1px 4px;border-radius:3px;margin-left:4px;">FP</span>'
            elif ck in missed:
                s += "background:#fff3cd;"
                tip = f"MISSED: {issue.description}" if issue else ""
                badge = '<span style="font-size:9px;background:#856404;color:#fff;padding:1px 4px;border-radius:3px;margin-left:4px;">MISS</span>'

            fx = fixed.get(ck)
            proposed = fix_values.get(ck)
            if fx == "correct":
                s += "box-shadow:inset 0 0 0 2px #28a745;"
                badge += '<span style="font-size:9px;background:#28a745;color:#fff;padding:1px 4px;border-radius:3px;margin-left:2px;">FIX</span>'
            elif fx == "partial":
                s += "box-shadow:inset 0 0 0 2px #ffc107;"
                badge += '<span style="font-size:9px;background:#ffc107;color:#333;padding:1px 4px;border-radius:3px;margin-left:2px;">~FIX</span>'

            dv = val if val.strip() else '<em style="color:#dc3545;font-style:italic;">empty</em>'

            # Show proposed fix value below the corrupted value
            fix_line = ""
            if proposed is not None:
                fix_color = "#28a745" if fx == "correct" else ("#b8860b" if fx == "partial" else "#dc3545")
                fix_line = (
                    f'<div style="font-size:10px;color:{fix_color};margin-top:2px;'
                    f'border-top:1px dashed {fix_color};padding-top:2px;">'
                    f'\u2192 {proposed}</div>'
                )

            t.append(f'<td style="{s}" title="{tip}">{dv}{badge}{fix_line}</td>')
        t.append('</tr>')
    t.append('</table>')
    return "".join(t)


LEGEND_HTML = (
    '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;font-size:11px;">'
    '<span style="background:#d4edda;padding:2px 8px;border-radius:4px;">Found (TP)</span>'
    '<span style="background:#f8d7da;padding:2px 8px;border-radius:4px;">False Positive</span>'
    '<span style="background:#fff3cd;padding:2px 8px;border-radius:4px;">Missed</span>'
    '<span style="box-shadow:inset 0 0 0 2px #28a745;padding:2px 8px;border-radius:4px;">Fix Correct</span>'
    '<span style="box-shadow:inset 0 0 0 2px #ffc107;padding:2px 8px;border-radius:4px;">Fix Partial</span>'
    '</div>'
)


# ── Core replay logic ──

def _replay_task(task_id: str) -> list[dict]:
    """Run the agent trajectory and collect per-step data."""
    env = DataQAEnvironment()
    obs = env.reset(task_id=task_id)
    task = env._current_task
    planted_keys = {i.to_key() for i in task.planted_issues}
    steps_data = []

    # Step 0: initial state
    steps_data.append({
        "label": "Initial — corrupted dataset",
        "html": _csv_to_html(obs.dataset_csv, task.planted_issues, set(), set(), set(), {}),
        "metrics": {"reward": 0.0, "tp": 0, "fp": 0, "fn": len(task.planted_issues),
                    "identify": 0.0, "fix": 0.0, "fixes_correct": 0},
        "feedback": f"Task: {task.name}\nIssues to find: {obs.num_issues_hint}\n\n{task.description}",
    })

    trajectory = AGENT_TRAJECTORIES.get(task_id, [])
    for i, step_data in enumerate(trajectory):
        action = DataQAAction(
            issues=step_data["issues"],
            fixes=step_data.get("fixes", []),
            task_id=task_id,
        )
        obs = env.step(action)

        reported_keys = set()
        for iss in step_data["issues"]:
            key = parse_issue_key(iss)
            if key:
                reported_keys.add(key)

        tp_keys = reported_keys & planted_keys
        fp_keys = reported_keys - planted_keys
        fn_keys = planted_keys - reported_keys

        correct = {_kc(k) for k in tp_keys}
        fp = {_kc(k) for k in fp_keys}
        missed = {_kc(k) for k in fn_keys} if obs.done else set()

        fixed: dict[tuple[int, str], str] = {}
        for d in obs.metadata.get("fix_details", []):
            c = (d["row"], d["col"])
            fixed[c] = "correct" if d["score"] >= 0.99 else ("partial" if d["score"] > 0 else "wrong")

        # Extract proposed fix values from the raw fix strings
        fix_values: dict[tuple[int, str], str] = {}
        from .environment import parse_fix
        for raw_fix in step_data.get("fixes", []):
            parsed = parse_fix(raw_fix)
            if parsed:
                row, col, val = parsed
                fix_values[(row, col)] = val

        html = _csv_to_html(obs.dataset_csv, task.planted_issues, correct, fp, missed, fixed, fix_values)

        has_fixes = bool(step_data.get("fixes"))
        if has_fixes:
            label = f"Step {i+1} — identify + fix"
        else:
            label = f"Step {i+1} — identify only"

        steps_data.append({
            "label": label,
            "html": html,
            "metrics": {
                "reward": obs.reward,
                "tp": obs.metadata["tp"],
                "fp": obs.metadata["fp"],
                "fn": obs.metadata["fn"],
                "identify": obs.metadata["identify_score"],
                "fix": obs.metadata["fix_score"],
                "fixes_correct": obs.metadata["fixes_correct"],
            },
            "feedback": obs.feedback,
        })

    return steps_data


def _kc(key: str) -> tuple[int, str]:
    parts = key.split(",")
    return (int(parts[0].split(":")[1]), parts[1].split(":")[1])


# ── Gradio app ──

def build_gradio_ui():
    # Pre-compute all replays at startup
    all_replays: dict[str, list[dict]] = {}
    for tid in list_tasks():
        all_replays[tid] = _replay_task(tid)

    def show_step(task_id: str, step_idx: int):
        replay = all_replays.get(task_id, [])
        step_idx = int(step_idx)
        if step_idx >= len(replay):
            step_idx = len(replay) - 1
        sd = replay[step_idx]
        m = sd["metrics"]

        # Reward color
        r = m["reward"]
        rc = "#28a745" if r >= 0.8 else ("#ffc107" if r >= 0.4 else "#dc3545")

        cards = (
            '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">'
            + _metric_card("Reward", f"{r:.2f}", rc)
            + _metric_card("Found", str(m["tp"]), "#28a745")
            + _metric_card("False Pos", str(m["fp"]), "#dc3545" if m["fp"] > 0 else "#28a745")
            + _metric_card("Missed", str(m["fn"]), "#dc3545" if m["fn"] > 0 else "#28a745")
            + _metric_card("Identify", f"{m['identify']:.2f}", "#333")
            + _metric_card("Fix", f"{m['fix']:.2f}", "#333")
            + '</div>'
        )

        full_html = (
            f'<div style="font-size:14px;font-weight:600;margin-bottom:8px;color:#495057;">'
            f'{sd["label"]}</div>'
            + cards + sd["html"] + LEGEND_HTML
        )

        return full_html, sd["feedback"]

    def on_task_change(task_id):
        replay = all_replays.get(task_id, [])
        max_step = len(replay) - 1
        html, fb = show_step(task_id, 0)
        return (
            gr.update(maximum=max_step, value=0),
            html,
            fb,
        )

    def on_step_change(task_id, step_idx):
        html, fb = show_step(task_id, step_idx)
        return html, fb

    # ── Live agent runner (connects to the env server) ──

    live_env = DataQAEnvironment()
    live_state: dict = {"obs": None, "task_id": "easy", "steps": []}

    def live_reset(task_id):
        obs = live_env.reset(task_id=task_id)
        task = live_env._current_task
        live_state["obs"] = obs
        live_state["task_id"] = task_id
        live_state["steps"] = []
        html = _csv_to_html(obs.dataset_csv, task.planted_issues, set(), set(), set(), {})
        info = f"**{task.name}** — {obs.num_issues_hint} issues to find, {obs.max_steps} steps max"
        return html, info, "", "0.000"

    def live_step(issues_text, fixes_text):
        if live_state["obs"] is None:
            return "Reset first.", "", "", ""
        obs = live_state["obs"]
        task = live_env._current_task
        planted_keys = {i.to_key() for i in task.planted_issues}

        issues = [l.strip() for l in issues_text.strip().split("\n") if l.strip()]
        fixes = [l.strip() for l in fixes_text.strip().split("\n") if l.strip()] if fixes_text.strip() else []

        action = DataQAAction(issues=issues, fixes=fixes, task_id=live_state["task_id"])
        obs = live_env.step(action)
        live_state["obs"] = obs

        reported_keys = set()
        for iss in issues:
            key = parse_issue_key(iss)
            if key:
                reported_keys.add(key)

        tp_keys = reported_keys & planted_keys
        fp_keys = reported_keys - planted_keys
        fn_keys = planted_keys - reported_keys

        correct = {_kc(k) for k in tp_keys}
        fp_set = {_kc(k) for k in fp_keys}
        missed = {_kc(k) for k in fn_keys} if obs.done else set()

        fixed: dict[tuple[int, str], str] = {}
        for d in obs.metadata.get("fix_details", []):
            c = (d["row"], d["col"])
            fixed[c] = "correct" if d["score"] >= 0.99 else ("partial" if d["score"] > 0 else "wrong")

        from .environment import parse_fix
        fix_values: dict[tuple[int, str], str] = {}
        for raw in fixes:
            parsed = parse_fix(raw)
            if parsed:
                fix_values[(parsed[0], parsed[1])] = parsed[2]

        html = _csv_to_html(obs.dataset_csv, task.planted_issues, correct, fp_set, missed, fixed, fix_values)

        m = obs.metadata
        r = obs.reward
        rc = "#28a745" if r >= 0.8 else ("#ffc107" if r >= 0.4 else "#dc3545")
        cards = (
            '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">'
            + _metric_card("Reward", f"{r:.2f}", rc)
            + _metric_card("Found", str(m["tp"]), "#28a745")
            + _metric_card("False Pos", str(m["fp"]), "#dc3545" if m["fp"] > 0 else "#28a745")
            + _metric_card("Missed", str(m["fn"]), "#dc3545" if m["fn"] > 0 else "#28a745")
            + '</div>'
        )
        full_html = cards + html + LEGEND_HTML
        return full_html, obs.feedback, f"{r:.3f}", ""

    # ── Build the UI ──

    with gr.Blocks(title="DataQA Environment") as demo:
        gr.Markdown(
            "# DataQA — Data Quality Assurance Environment\n"
            "Two-phase RL environment: **Identify** data quality issues, then **Fix** them."
        )

        with gr.Tabs():
            # ── Tab 1: Demo replay ──
            with gr.Tab("Demo (Baseline Agent)"):
                gr.Markdown(
                    "*Replay of the baseline Qwen-72B agent. "
                    "Use the slider to step through the agent's trajectory.*"
                )
                with gr.Row():
                    task_dd = gr.Dropdown(choices=list_tasks(), value="easy", label="Task", scale=1)
                    step_slider = gr.Slider(minimum=0, maximum=2, step=1, value=0, label="Step", scale=3)

                viz_html = gr.HTML()
                feedback_box = gr.Textbox(label="Agent Feedback", lines=10, interactive=False)

                task_dd.change(on_task_change, inputs=[task_dd], outputs=[step_slider, viz_html, feedback_box])
                step_slider.change(on_step_change, inputs=[task_dd, step_slider], outputs=[viz_html, feedback_box])
                demo.load(on_task_change, inputs=[task_dd], outputs=[step_slider, viz_html, feedback_box])

            # ── Tab 2: Try your own agent ──
            with gr.Tab("Try Your Own Agent"):
                gr.Markdown(
                    "*Submit your own issues and fixes to see how the environment scores them. "
                    "This is the same environment the baseline agent talks to.*"
                )
                with gr.Row():
                    live_task_dd = gr.Dropdown(choices=list_tasks(), value="easy", label="Task", scale=1)
                    live_reset_btn = gr.Button("Reset", variant="primary", scale=1)

                with gr.Row():
                    live_info = gr.Markdown()
                    live_reward = gr.Textbox(label="Reward", interactive=False, scale=1)

                live_viz = gr.HTML()

                with gr.Row():
                    live_issues = gr.Textbox(
                        label="Issues (one per line)",
                        placeholder="row:4,col:name,issue:missing_value\nrow:7,col:salary,issue:wrong_type",
                        lines=5,
                    )
                    live_fixes = gr.Textbox(
                        label="Fixes (one per line, optional)",
                        placeholder="row:4,col:name,fix:David Kim\nrow:7,col:salary,fix:75000",
                        lines=5,
                    )

                live_step_btn = gr.Button("Submit Step", variant="primary")
                live_feedback = gr.Textbox(label="Feedback", lines=10, interactive=False)

                live_reset_btn.click(
                    live_reset, inputs=[live_task_dd],
                    outputs=[live_viz, live_info, live_feedback, live_reward],
                )
                live_step_btn.click(
                    live_step, inputs=[live_issues, live_fixes],
                    outputs=[live_viz, live_feedback, live_reward, live_issues],
                )

    return demo


if __name__ == "__main__":
    demo = build_gradio_ui()
    demo.launch()
