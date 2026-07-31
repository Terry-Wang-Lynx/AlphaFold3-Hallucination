"""`af3h` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

from .audit import audit_run
from .config import load_config
from .doctor import environment_report
from .errors import AF3HallucinationError
from .hallucination import run_hallucination
from .plugins import default_registry
from .schedule import SCHEDULE_PROVENANCE, expand_schedule
from .workflow import AntibodyWorkflow


def _json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="af3h",
        description="AlphaFold3 Hallucination research framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="inspect the current runtime")
    doctor.add_argument("--require-jax", action="store_true")
    doctor.add_argument("--require-af3", action="store_true")

    config = sub.add_parser("config", help="configuration tools")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    validate = config_sub.add_parser("validate", help="validate and resolve YAML")
    validate.add_argument("path")

    schedule = sub.add_parser("schedule", help="Hallucination schedule tools")
    schedule_sub = schedule.add_subparsers(dest="schedule_command", required=True)
    show = schedule_sub.add_parser("show", help="expand all per-step parameters")
    show.add_argument("path")

    plugins = sub.add_parser("plugins", help="plugin tools")
    plugins_sub = plugins.add_subparsers(dest="plugins_command", required=True)
    plugins_sub.add_parser("list", help="list built-in plugins")

    audit = sub.add_parser("audit", help="verify hashes and run-state integrity")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    audit_run_parser = audit_sub.add_parser("run", help="audit a completed run directory")
    audit_run_parser.add_argument("path")

    hallucinate = sub.add_parser("hallucinate", help="run AF3/JAX Hallucination")
    hallucinate_sub = hallucinate.add_subparsers(dest="hallucinate_command", required=True)
    run = hallucinate_sub.add_parser("run")
    run.add_argument("config")
    run.add_argument("--input-json")
    run.add_argument("--model-dir")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--dry-run", action="store_true")

    antibody = sub.add_parser("antibody", help="antibody workflow")
    antibody_sub = antibody.add_subparsers(dest="antibody_command", required=True)
    plan = antibody_sub.add_parser("plan", help="show the resolved five-stage workflow")
    plan.add_argument("config")
    antibody_run = antibody_sub.add_parser("run", help="execute or resume a workflow")
    antibody_run.add_argument("config")
    antibody_run.add_argument("--output-dir", required=True)
    antibody_run.add_argument("--resume", action="store_true")

    pocket = sub.add_parser("pocket", help="small-molecule pocket placeholder")
    pocket_sub = pocket.add_subparsers(dest="pocket_command", required=True)
    pocket_sub.add_parser("status")
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        report = environment_report()
        _json(report)
        if args.require_jax and not report["capabilities"]["jax"]:
            return 3
        if args.require_af3 and not report["capabilities"]["af3"]:
            return 4
        return 0
    if args.command == "config":
        config = load_config(args.path)
        _json({"status": "valid", "config_sha256": config.sha256, "config": config.to_dict()})
        return 0
    if args.command == "schedule":
        config = load_config(args.path)
        _json({"provenance": SCHEDULE_PROVENANCE, "steps": expand_schedule(config.hallucination)})
        return 0
    if args.command == "plugins":
        _json(default_registry().inventory())
        return 0
    if args.command == "audit":
        result = audit_run(args.path)
        _json(result)
        return 0 if result["status"] == "pass" else 5
    if args.command == "hallucinate":
        config = load_config(args.config)
        summary = run_hallucination(
            config,
            input_json=args.input_json,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        _json(summary)
        return 0
    if args.command == "antibody":
        config = load_config(args.config)
        workflow = AntibodyWorkflow(config)
        if args.antibody_command == "plan":
            _json({"config_sha256": config.sha256, "steps": workflow.plan()})
            return 0
        state = workflow.run(args.output_dir, resume=args.resume)
        _json(state)
        return 0
    if args.command == "pocket":
        _json(
            {
                "status": "not_implemented",
                "module": "small_molecule_pocket_redesign",
                "message": "Placeholder only; no pocket-redesign functionality is shipped.",
            }
        )
        return 0
    raise RuntimeError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    original_argv = sys.argv
    sys.argv = [original_argv[0]]
    try:
        try:
            return _run(args)
        except (AF3HallucinationError, FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"af3h: {exc}", file=sys.stderr)
            return 2
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
