"""Command-line interface; help and demo never import optional vision packages."""

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path

from .config import load_config
from .demo import run_demo
from .reporting import write_report

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart crosswalk video-analysis prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run deterministic synthetic tracks without model files")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    check = commands.add_parser("check", help="Validate configuration and check local asset paths")
    check.add_argument("--config", type=Path, required=True)
    analyze = commands.add_parser("analyze", help="Analyze a local video using local YOLO weights")
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--input", type=Path, help="Override the configured video path")
    analyze.add_argument("--detector", type=Path)
    analyze.add_argument("--classifier", type=Path)
    analyze.add_argument("--stride", type=int)
    analyze.add_argument("--device", help="Ultralytics device, e.g. cpu, mps or 0")
    analyze.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    analyze.add_argument("--save-video", action="store_true", help="Export annotated sampled frames")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "demo":
            report = run_demo()
        else:
            config = load_config(args.config)
            if args.command == "check":
                paths = {key: getattr(config, key) for key in ("video", "detector", "classifier")}
                checks = {key: {"path": str(path), "exists": path.is_file()} for key, path in paths.items()}
                print(json.dumps({"configuration": "valid", "assets": checks}, ensure_ascii=False, indent=2))
                return 0 if all(item["exists"] for item in checks.values()) else 1
            overrides = {key: getattr(args, arg).resolve() for key, arg in
                         (("video", "input"), ("detector", "detector"), ("classifier", "classifier"))
                         if getattr(args, arg) is not None}
            tracking_overrides = {key: getattr(args, key) for key in ("stride", "device")
                                  if getattr(args, key) is not None}
            config = replace(config, **overrides, tracking=replace(config.tracking, **tracking_overrides))
            from .pipeline import run_analysis
            annotated_output = args.output / "annotated.mp4" if args.save_video else None
            report = run_analysis(config, annotated_output=annotated_output)
        summary, signal = write_report(report, args.output)
        decision = report["decision"]
        LOGGER.info("Additional time: %s s; controlling track: %s", decision["additional_time_s"],
                    decision["controlling_track_id"])
        LOGGER.info("Saved %s and %s", summary, signal)
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
