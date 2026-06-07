"""Build an interactive HTML replay from ToM JSONL logs and saved frames.

Usage:
    python -m viz.replay_html logs/tom/tom_session_20260510_183018.jsonl \
        --frames logs/frames
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"step_(\d+)\.png$")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("self_role") in ("evader", "defender"):
                rows.append(row)
    rows.sort(key=lambda r: (int(r.get("step", 0)), str(r.get("self_role", ""))))
    return rows


def load_frames(frames_dir: Path, out_dir: Path, embed: bool = False) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for path in sorted(frames_dir.glob("step_*.png")):
        match = STEP_RE.match(path.name)
        if not match:
            continue
        if embed:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            img_path = f"data:image/png;base64,{encoded}"
        else:
            img_path = os.path.relpath(path.resolve(), out_dir.resolve())
        frames.append({"step": int(match.group(1)), "path": img_path})
    return frames


def nearest_frame(step: int, frames: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not frames:
        return None
    return min(frames, key=lambda f: abs(int(f["step"]) - step))


def belief(row: dict[str, Any]) -> dict[str, Any]:
    l2 = row.get("l2") or {}
    b = l2.get("opponent_belief") or {}
    return {
        "p_goal_A": b.get("p_goal_A"),
        "p_goal_B": b.get("p_goal_B"),
        "p_undecided": b.get("p_undecided"),
        "commitment_evidence": l2.get("commitment_evidence"),
        "rationale": l2.get("rationale", ""),
    }


def compact_role(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("state") or {}
    waypoint = row.get("waypoint") or {}
    l1 = row.get("l1") or {}
    return {
        "role": row.get("self_role"),
        "state": {
            "self": state.get("self") or {},
            "opponent": state.get("opponent") or {},
        },
        "waypoint": {
            "target_xy": waypoint.get("target_xy"),
            "raw_target_xy": waypoint.get("raw_target_xy"),
            "speed_frac": waypoint.get("speed_frac"),
            "phase": waypoint.get("phase"),
            "clamped_horizon": waypoint.get("clamped_horizon"),
            "rationale": waypoint.get("rationale", ""),
        },
        "l1": {
            "opponent_intent": l1.get("opponent_intent"),
            "confidence": l1.get("confidence"),
            "rationale": l1.get("rationale", ""),
            "cues": l1.get("cues") or [],
        },
        "l2": belief(row),
    }


def build_decisions(rows: list[dict[str, Any]], frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = int(row.get("step", 0))
        state = row.get("state") or {}
        entry = grouped.setdefault(step, {
            "step": step,
            "sim_time_sec": state.get("sim_time_sec"),
            "roles": {},
            "goals": {
                "A": state.get("goal_A_xy", [-8.0, 20.0]),
                "B": state.get("goal_B_xy", [8.0, 20.0]),
            },
        })
        if entry.get("sim_time_sec") is None:
            entry["sim_time_sec"] = state.get("sim_time_sec")
        entry["roles"][str(row.get("self_role"))] = compact_role(row)

    decisions = [grouped[k] for k in sorted(grouped)]
    for decision in decisions:
        decision["frame"] = nearest_frame(int(decision["step"]), frames)
    return decisions


def html_document(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ToM Replay</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f8fb;
  --panel: #ffffff;
  --ink: #16202a;
  --muted: #657282;
  --line: #d9dee7;
  --evader: #1f77b4;
  --defender: #d62728;
  --goal-a: #219653;
  --goal-b: #f2994a;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); }}
.app {{ display: grid; grid-template-rows: auto 1fr auto; min-height: 100vh; }}
header {{
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 14px 18px; border-bottom: 1px solid var(--line); background: var(--panel);
}}
h1 {{ margin: 0; font-size: 18px; font-weight: 700; }}
.meta {{ color: var(--muted); font-size: 13px; }}
.main {{
  display: grid; grid-template-columns: minmax(420px, 1.25fr) minmax(360px, 0.75fr);
  gap: 14px; padding: 14px; min-height: 0;
}}
.panel {{
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}}
.viewer {{ display: grid; grid-template-rows: auto 1fr auto; min-height: 0; overflow: hidden; }}
.toolbar {{ display: grid; grid-template-columns: auto 1fr auto auto; gap: 12px; align-items: center; padding: 12px; border-bottom: 1px solid var(--line); }}
button {{
  border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink);
  padding: 7px 10px; font-weight: 650; cursor: pointer;
}}
button:hover {{ background: #f1f4f8; }}
input[type="range"] {{ width: 100%; }}
.frameWrap {{ min-height: 0; display: grid; place-items: center; background: #eef2f6; padding: 12px; }}
#frame {{ max-width: 100%; max-height: 66vh; object-fit: contain; border-radius: 6px; border: 1px solid var(--line); background: white; }}
.caption {{ padding: 10px 12px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }}
.side {{ display: grid; grid-template-rows: auto 1fr; min-height: 0; }}
.cards {{ overflow: auto; padding: 12px; display: grid; gap: 12px; }}
.roleCard {{ border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
.roleHead {{ display: flex; align-items: center; justify-content: space-between; padding: 9px 10px; font-weight: 750; color: white; }}
.roleHead.evader {{ background: var(--evader); }}
.roleHead.defender {{ background: var(--defender); }}
.roleBody {{ padding: 10px; display: grid; gap: 10px; }}
.kv {{ display: grid; grid-template-columns: 115px 1fr; gap: 7px; font-size: 13px; }}
.kv .k {{ color: var(--muted); }}
.chipRow {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border: 1px solid var(--line); border-radius: 999px; background: #f8fafc; font-size: 12px; }}
details {{ border-top: 1px solid var(--line); padding-top: 8px; }}
summary {{ cursor: pointer; color: var(--muted); font-size: 13px; font-weight: 650; }}
.rationale {{ margin: 8px 0 0; color: #2f3b4a; font-size: 12px; line-height: 1.45; white-space: pre-wrap; }}
.timelinePanel {{ margin: 0 14px 14px; padding: 12px; }}
#timeline {{ width: 100%; height: 220px; display: block; }}
.empty {{ color: var(--muted); padding: 18px; }}
@media (max-width: 980px) {{
  .main {{ grid-template-columns: 1fr; }}
  #frame {{ max-height: 54vh; }}
}}
</style>
</head>
<body>
<div class="app">
  <header>
    <div>
      <h1>ToM Replay</h1>
      <div class="meta" id="sessionMeta"></div>
    </div>
    <div class="meta">Slider = decision event, not every physics step</div>
  </header>

  <main class="main">
    <section class="panel viewer">
      <div class="toolbar">
        <button id="prevBtn" type="button">Prev</button>
        <input id="slider" type="range" min="0" max="0" value="0">
        <button id="nextBtn" type="button">Next</button>
        <div class="meta" id="stepLabel"></div>
      </div>
      <div class="frameWrap">
        <img id="frame" alt="Replay frame">
      </div>
      <div class="caption" id="frameCaption"></div>
    </section>

    <aside class="panel side">
      <div class="toolbar" style="grid-template-columns:1fr auto;">
        <strong>Decision Panel</strong>
        <span class="meta" id="timeLabel"></span>
      </div>
      <div class="cards" id="cards"></div>
    </aside>
  </main>

  <section class="panel timelinePanel">
    <svg id="timeline" role="img" aria-label="Belief and phase timeline"></svg>
  </section>
</div>

<script id="payload" type="application/json">{data}</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
const decisions = payload.decisions || [];
const roles = ['evader', 'defender'];
const roleColor = {{ evader: '#1f77b4', defender: '#d62728' }};
const phaseColor = {{
  shaping: '#7b61ff', exploit: '#d9480f', exploiting: '#d9480f',
  holding: '#6c757d', tracking: '#0ca678', committing: '#e03131',
  observing: '#868e96'
}};

const slider = document.getElementById('slider');
const frame = document.getElementById('frame');
const cards = document.getElementById('cards');
const sessionMeta = document.getElementById('sessionMeta');
const stepLabel = document.getElementById('stepLabel');
const timeLabel = document.getElementById('timeLabel');
const frameCaption = document.getElementById('frameCaption');

slider.max = Math.max(0, decisions.length - 1);
sessionMeta.textContent = `${{payload.session}} · ${{decisions.length}} decision ticks · ${{payload.frames_count}} frames`;

function fmtNum(v, digits = 2) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return 'n/a';
  return Number(v).toFixed(digits);
}}

function fmtXY(v) {{
  if (!Array.isArray(v) || v.length !== 2) return 'n/a';
  return `(${{fmtNum(v[0])}}, ${{fmtNum(v[1])}})`;
}}

function beliefChips(l2) {{
  return `
    <span class="chip">A ${{fmtNum(l2?.p_goal_A)}}</span>
    <span class="chip">B ${{fmtNum(l2?.p_goal_B)}}</span>
    <span class="chip">undecided ${{fmtNum(l2?.p_undecided)}}</span>
    <span class="chip">commit ${{fmtNum(l2?.commitment_evidence)}}</span>
  `;
}}

function roleCard(role, data) {{
  if (!data) {{
    return `<div class="roleCard"><div class="roleHead ${{role}}">${{role.toUpperCase()}}</div><div class="empty">No decision row for this role at this step.</div></div>`;
  }}
  const wp = data.waypoint || {{}};
  const l1 = data.l1 || {{}};
  const l2 = data.l2 || {{}};
  const self = data.state?.self || {{}};
  const phase = wp.phase || 'n/a';
  return `
    <div class="roleCard">
      <div class="roleHead ${{role}}">
        <span>${{role.toUpperCase()}}</span>
        <span>${{phase}}</span>
      </div>
      <div class="roleBody">
        <div class="kv"><div class="k">pos</div><div>${{fmtXY(self.pos_xy)}} · heading ${{fmtNum(self.heading_deg, 1)}}°</div></div>
        <div class="kv"><div class="k">waypoint</div><div>${{fmtXY(wp.target_xy)}} · speed ${{fmtNum(wp.speed_frac)}}</div></div>
        <div class="kv"><div class="k">L1 opp intent</div><div>${{l1.opponent_intent || 'n/a'}} · confidence ${{fmtNum(l1.confidence)}}</div></div>
        <div class="kv"><div class="k">L2 opp belief</div><div>A=${{fmtNum(l2?.p_goal_A)}} · B=${{fmtNum(l2?.p_goal_B)}} · undecided=${{fmtNum(l2?.p_undecided)}}</div></div>
        <div class="chipRow">${{beliefChips(l2)}}</div>
        <details>
          <summary>Waypoint rationale</summary>
          <p class="rationale">${{escapeHtml(wp.rationale || '')}}</p>
        </details>
        <details>
          <summary>L1/L2 rationale</summary>
          <p class="rationale">${{escapeHtml((l1.rationale || '') + '\\n\\n' + (l2.rationale || ''))}}</p>
        </details>
      </div>
    </div>
  `;
}}

function escapeHtml(text) {{
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}}

function render(i) {{
  if (!decisions.length) {{
    cards.innerHTML = '<div class="empty">No decision rows found.</div>';
    return;
  }}
  i = Math.max(0, Math.min(decisions.length - 1, i));
  slider.value = i;
  const d = decisions[i];
  stepLabel.textContent = `#${{i + 1}} / ${{decisions.length}}`;
  timeLabel.textContent = `step ${{d.step}} · t=${{fmtNum(d.sim_time_sec)}}s`;
  if (d.frame?.path) {{
    frame.src = d.frame.path;
    frame.style.display = 'block';
    frameCaption.textContent = `Showing nearest saved frame: step ${{d.frame.step}}`;
  }} else {{
    frame.removeAttribute('src');
    frame.style.display = 'none';
    frameCaption.textContent = 'No saved frame found.';
  }}
  cards.innerHTML = roles.map(role => roleCard(role, d.roles?.[role])).join('');
  drawTimeline(i);
}}

function timeAt(d) {{
  return Number(d.sim_time_sec ?? d.step ?? 0);
}}

function pointFor(d, role, key) {{
  const v = d.roles?.[role]?.l2?.[key];
  return v === null || v === undefined ? null : Number(v);
}}

function drawTimeline(activeIndex) {{
  const svg = document.getElementById('timeline');
  const w = svg.clientWidth || 900;
  const h = svg.clientHeight || 220;
  svg.setAttribute('viewBox', `0 0 ${{w}} ${{h}}`);
  svg.innerHTML = '';
  if (!decisions.length) return;

  const padL = 54, padR = 16, padT = 18, rowH = 82, gap = 18;
  const t0 = timeAt(decisions[0]);
  const t1 = timeAt(decisions[decisions.length - 1]);
  const span = Math.max(1e-9, t1 - t0);
  const x = d => padL + ((timeAt(d) - t0) / span) * (w - padL - padR);
  const y = (row, v) => padT + row * (rowH + gap) + (1 - Math.max(0, Math.min(1, v))) * 56 + 16;

  const add = (tag, attrs = {{}}, text = '') => {{
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (text) el.textContent = text;
    svg.appendChild(el);
    return el;
  }};

  roles.forEach((role, row) => {{
    const top = padT + row * (rowH + gap);
    add('text', {{x: 0, y: top + 42, fill: roleColor[role], 'font-size': 12, 'font-weight': 700}}, role);
    add('line', {{x1: padL, y1: top + 72, x2: w - padR, y2: top + 72, stroke: '#d9dee7'}});
    add('line', {{x1: padL, y1: top + 16, x2: padL, y2: top + 72, stroke: '#d9dee7'}});
    [['p_goal_A', '#219653'], ['p_goal_B', '#f2994a'], ['p_undecided', '#657282']].forEach(([key, color]) => {{
      let path = '';
      decisions.forEach(d => {{
        const v = pointFor(d, role, key);
        if (v === null || Number.isNaN(v)) return;
        path += `${{path ? 'L' : 'M'}}${{x(d).toFixed(1)}} ${{y(row, v).toFixed(1)}} `;
      }});
      if (path) add('path', {{d: path, fill: 'none', stroke: color, 'stroke-width': 2}});
    }});
    decisions.forEach(d => {{
      const phase = d.roles?.[role]?.waypoint?.phase;
      if (!phase) return;
      add('circle', {{cx: x(d), cy: top + 8, r: 4, fill: phaseColor[phase] || roleColor[role], opacity: 0.9}});
    }});
  }});

  const cursorX = x(decisions[activeIndex]);
  add('line', {{x1: cursorX, y1: 8, x2: cursorX, y2: h - 10, stroke: '#111827', 'stroke-width': 1.5, 'stroke-dasharray': '4 4'}});
  add('text', {{x: padL, y: h - 6, fill: '#219653', 'font-size': 11}}, 'A');
  add('text', {{x: padL + 22, y: h - 6, fill: '#f2994a', 'font-size': 11}}, 'B');
  add('text', {{x: padL + 44, y: h - 6, fill: '#657282', 'font-size': 11}}, 'undecided');
  add('text', {{x: w - 120, y: h - 6, fill: '#657282', 'font-size': 11}}, `t=${{fmtNum(timeAt(decisions[activeIndex]))}}s`);
}}

document.getElementById('prevBtn').addEventListener('click', () => render(Number(slider.value) - 1));
document.getElementById('nextBtn').addEventListener('click', () => render(Number(slider.value) + 1));
slider.addEventListener('input', () => render(Number(slider.value)));
window.addEventListener('resize', () => drawTimeline(Number(slider.value)));
window.addEventListener('keydown', event => {{
  if (event.key === 'ArrowLeft') render(Number(slider.value) - 1);
  if (event.key === 'ArrowRight') render(Number(slider.value) + 1);
}});
render(0);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--frames", type=Path, default=Path("logs/frames"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--link-frames", action="store_true",
                        help="Link PNG frames by relative path instead of embedding them in the HTML.")
    args = parser.parse_args()

    session = args.session.resolve()
    frames_dir = args.frames.resolve()
    out = (args.out or session.with_name(f"{session.stem}_replay.html")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(session)
    frames = load_frames(frames_dir, out.parent, embed=not args.link_frames)
    decisions = build_decisions(rows, frames)
    payload = {
        "session": session.name,
        "decisions": decisions,
        "frames_count": len(frames),
    }
    out.write_text(html_document(payload), encoding="utf-8")
    print(f"wrote {out} ({len(decisions)} decision ticks, {len(frames)} frames)")


if __name__ == "__main__":
    main()
