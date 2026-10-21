"""Write a readable JSON report and the prototype's two-value signal file."""

import json
from pathlib import Path


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.json"
    signal = output_dir / "signal.txt"
    decision = report["decision"]
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    signal.write_text(f"{decision['controlling_track_id']} {decision['additional_time_s']}\n", encoding="utf-8")
    return summary, signal
