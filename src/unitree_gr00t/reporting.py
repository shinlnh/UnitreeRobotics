"""Self-contained JSON/JSONL/HTML reporting for generalist evaluations."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .types import EpisodeResult


def _scene_svg(result: EpisodeResult) -> str:
    shapes: list[str] = []
    for name, obj in result.final_objects.items():
        x = 28 + float(obj["x"]) * 344
        y = 24 + (1.0 - float(obj["y"])) * 212
        color = html.escape(str(obj["color"]))
        label = html.escape(name.replace("_", " "))
        if obj["kind"] == "bin":
            shapes.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="24" fill="{color}" fill-opacity=".22" '
                f'stroke="{color}" stroke-width="4"/><text x="{x:.1f}" y="{y + 38:.1f}" '
                f'text-anchor="middle">{label}</text>'
            )
        else:
            shapes.append(
                f'<rect x="{x - 11:.1f}" y="{y - 11:.1f}" width="22" height="22" rx="4" '
                f'fill="{color}"/><text x="{x:.1f}" y="{y + 27:.1f}" text-anchor="middle">{label}</text>'
            )
    return (
        '<svg viewBox="0 0 400 260" role="img" aria-label="Final tabletop state">'
        '<rect x="4" y="4" width="392" height="252" rx="16" fill="#111827" stroke="#334155"/>'
        + "".join(shapes)
        + "</svg>"
    )


def write_report(
    results: list[EpisodeResult],
    output_dir: str | Path,
    backend: str = "mock",
) -> dict[str, object]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    total = len(results)
    successes = sum(result.success for result in results)
    summary: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "backend": backend,
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "mean_steps": sum(result.steps for result in results) / total if total else 0.0,
        "safety_clips": sum(result.safety_clips for result in results),
        "report": str(directory / "report.html"),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (directory / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    rows: list[str] = []
    cards: list[str] = []
    for result in results:
        status = "PASS" if result.success else "FAIL"
        status_class = "pass" if result.success else "fail"
        rows.append(
            "<tr>"
            f'<td><span class="badge {status_class}">{status}</span></td>'
            f"<td>{html.escape(result.task_id)}</td>"
            f"<td>{result.seed}</td><td>{result.steps}</td><td>{result.policy_calls}</td>"
            f"<td>{result.safety_clips}</td><td>{result.elapsed_seconds * 1000:.1f} ms</td>"
            "</tr>"
        )
        cards.append(
            '<article class="episode">'
            f"<h3>{html.escape(result.task_id)}</h3>"
            f"<p>{html.escape(result.instruction)}</p>{_scene_svg(result)}"
            f'<p class="events">{html.escape(" → ".join(result.events))}</p>'
            "</article>"
        )

    success_percent = float(summary["success_rate"]) * 100
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Unitree G1 GR00T generalist report</title>
<style>
:root{{--bg:#07111f;--panel:#0f1c2e;--line:#25354d;--text:#e5edf7;--muted:#91a4bd;--ok:#34d399;--bad:#fb7185}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at top,#142742,var(--bg) 42%);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:48px 24px}} h1{{font-size:clamp(30px,5vw,56px);line-height:1;margin:.2em 0}} .eyebrow{{color:#60a5fa;letter-spacing:.14em;text-transform:uppercase;font-weight:700}}
.lede,.events{{color:var(--muted)}} .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:28px 0}}
.metric,.episode,.table-wrap{{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 18px 50px #0004}}
.metric strong{{display:block;font-size:32px}} table{{width:100%;border-collapse:collapse}} th,td{{padding:12px 10px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--muted)}} .table-wrap{{overflow:auto}}
.badge{{font-weight:800;font-size:12px;padding:5px 9px;border-radius:99px}} .pass{{color:var(--ok);background:#064e3b66}} .fail{{color:var(--bad);background:#88133766}}
.episodes{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px;margin-top:18px}} .episode svg{{width:100%;height:auto}} .episode svg text{{fill:#cbd5e1;font-size:10px}}
code{{color:#93c5fd}} footer{{color:var(--muted);margin-top:28px}}
</style></head><body><main>
<p class="eyebrow">GR00T N1.7 · UNITREE_G1_SONIC · evaluation harness</p>
<h1>Generalist robotics smoke report</h1>
<p class="lede">Backend <code>{html.escape(backend)}</code>. This CPU smoke backend validates task orchestration, action chunks, safety gating, rollout and reporting; it is not a learned GR00T checkpoint.</p>
<section class="metrics"><div class="metric"><span>Success rate</span><strong>{success_percent:.1f}%</strong></div><div class="metric"><span>Episodes</span><strong>{total}</strong></div><div class="metric"><span>Mean steps</span><strong>{float(summary["mean_steps"]):.1f}</strong></div><div class="metric"><span>Safety clips</span><strong>{summary["safety_clips"]}</strong></div></section>
<section class="table-wrap"><table><thead><tr><th>Result</th><th>Task</th><th>Seed</th><th>Steps</th><th>Policy calls</th><th>Clips</th><th>Runtime</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>
<h2>Final scene per episode</h2><section class="episodes">{"".join(cards)}</section>
<footer>Generated {html.escape(str(summary["generated_at"]))}</footer></main></body></html>"""
    (directory / "report.html").write_text(page, encoding="utf-8")
    return summary
