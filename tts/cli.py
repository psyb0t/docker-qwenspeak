"""Qwen3-TTS: YAML-driven text-to-speech pipeline.

Pipe a YAML config via stdin to submit an async TTS job.
Returns a job UUID immediately. Poll with get-job.

Usage:
  tts                    Read YAML from stdin, submit async job
  tts print-yaml         Print a template YAML config to stdout
  tts list-speakers      List available preset speakers
  tts list-jobs          List all TTS jobs
  tts get-job <id>       Get job details as JSON
  tts get-job-log <id>   View job log (with -f to follow)
  tts cancel-job <id>    Cancel a running/queued job
  tts log [-f] [-n N]    View TTS logs
  tts tokenize <audio>   Encode/decode audio through the speech tokenizer

Examples:
  # Submit a job
  ssh tts@host "tts" < job.yaml
  # {"id": "...", "status": "queued", ...}

  # Check progress
  ssh tts@host "tts get-job 550e8400"

  # List all jobs
  ssh tts@host "tts list-jobs"

  # Cancel
  ssh tts@host "tts cancel-job 550e8400"

  # View logs
  ssh tts@host "tts log -f"
"""

import argparse
import json
import os
import sys

import yaml

from tts.commands import (
    cmd_cancel_job,
    cmd_get_job,
    cmd_get_job_log,
    cmd_list_jobs,
    cmd_list_speakers,
    cmd_log,
    cmd_tokenize,
)
from tts.constants import DEFAULT_MODELS_DIR, VALID_MODES, YAML_TEMPLATE
from tts.jobs import count_active_jobs, create_job, max_queue
from tts.pipeline import daemonize, run_worker
from tts.utils import die


def validate_config(config: dict) -> None:
    """Validate YAML config before creating a job. Dies on errors."""
    if not isinstance(config, dict):
        die("YAML config must be a mapping")

    steps = config.get("steps")
    if not steps or not isinstance(steps, list):
        die("YAML config must have a 'steps' list")

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            die(f"Step {i + 1} must be a mapping")

        mode = step.get("mode")
        if mode not in VALID_MODES:
            die(f"Step {i + 1}: invalid mode '{mode}'. Choose from: {VALID_MODES}")

        generations = step.get("generate")
        if not generations or not isinstance(generations, list):
            die(f"Step {i + 1}: mode '{mode}' has no 'generate' list")

        for j, gen in enumerate(generations):
            if not isinstance(gen, dict):
                die(f"Step {i + 1}, generation {j + 1}: must be a mapping")
            if not gen.get("text"):
                die(f"Step {i + 1}, generation {j + 1}: missing 'text'")
            if not gen.get("output"):
                die(f"Step {i + 1}, generation {j + 1}: missing 'output'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS: YAML-driven text-to-speech pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models-dir",
        "-m",
        default=DEFAULT_MODELS_DIR,
        help=f"Base directory for models (default: {DEFAULT_MODELS_DIR})",
    )

    sub = parser.add_subparsers(dest="command")

    p_py = sub.add_parser("print-yaml", help="Print a template YAML config to stdout")
    p_py.set_defaults(func=lambda _: print(YAML_TEMPLATE))

    p_ls = sub.add_parser("list-speakers", help="List available preset speakers")
    p_ls.set_defaults(func=lambda _: cmd_list_speakers())

    p_lj = sub.add_parser("list-jobs", help="List all TTS jobs")
    p_lj.add_argument("--json", action="store_true", default=False, help="JSON output")
    p_lj.set_defaults(func=cmd_list_jobs)

    p_gj = sub.add_parser("get-job", help="Get job details as JSON")
    p_gj.add_argument("id", help="Job UUID (or prefix)")
    p_gj.set_defaults(func=cmd_get_job)

    p_cj = sub.add_parser("cancel-job", help="Cancel a running/queued job")
    p_cj.add_argument("id", help="Job UUID (or prefix)")
    p_cj.set_defaults(func=cmd_cancel_job)

    p_gjl = sub.add_parser("get-job-log", help="View job log output")
    p_gjl.add_argument("id", help="Job UUID (or prefix)")
    p_gjl.add_argument(
        "-f",
        "--follow",
        action="store_true",
        default=False,
        help="Follow log output (like tail -f)",
    )
    p_gjl.add_argument("-n", type=int, default=20, help="Number of lines to show (default: 20)")
    p_gjl.set_defaults(func=cmd_get_job_log)

    p_log = sub.add_parser("log", help="View TTS logs")
    p_log.add_argument(
        "-f",
        "--follow",
        action="store_true",
        default=False,
        help="Follow log output (like tail -f)",
    )
    p_log.add_argument("-n", type=int, default=20, help="Number of lines to show (default: 20)")
    p_log.set_defaults(func=cmd_log)

    p_tok = sub.add_parser("tokenize", help="Encode audio to speech tokens and decode back")
    p_tok.add_argument("audio", help="Input audio file path")
    p_tok.add_argument(
        "--output",
        "-o",
        default="output.wav",
        help="Output file path (default: output.wav)",
    )
    p_tok.add_argument(
        "--tokenizer",
        default="12hz",
        choices=["12hz"],
        help="Tokenizer variant (default: 12hz)",
    )
    p_tok.set_defaults(func=cmd_tokenize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command:
        args.func(args)
        return

    # No subcommand → read YAML from stdin, submit async job
    if sys.stdin.isatty():
        parser.print_help()
        sys.exit(1)

    raw = sys.stdin.read()
    try:
        config = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        die(f"Invalid YAML: {e}")

    validate_config(config)

    # Check queue capacity
    active = count_active_jobs()
    limit = max_queue()
    if active >= limit:
        die(f"Queue full ({active}/{limit}). Try again later or cancel a job.")

    job = create_job(config)

    pid = os.fork()
    if pid > 0:
        # Parent: print job summary, exit immediately
        print(
            json.dumps(
                {
                    "id": job["id"],
                    "status": job["status"],
                    "total_steps": job["progress"]["total_steps"],
                    "total_generations": job["progress"]["total_generations"],
                }
            )
        )
        sys.exit(0)

    # Child: become daemon, then run worker loop
    daemonize()
    run_worker()
