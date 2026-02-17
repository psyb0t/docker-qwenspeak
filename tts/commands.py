from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from tts.constants import DEVICE, JOBS_DIR, LOG_DIR, SPEAKER_INFO
from tts.jobs import (
    _job_log_path,
    _pid_alive,
    _resolve_job_id,
    load_job,
    save_job,
)
from tts.logging import setup_logging
from tts.utils import _tail, die, resolve_tokenizer_path, safe_path, save_audio


def cmd_list_speakers() -> None:
    print("Available speakers for CustomVoice models:")
    print()
    for name, (gender, lang, desc) in SPEAKER_INFO.items():
        print(f"  {name:<12} {gender:<8} {lang:<10} {desc}")


def cmd_list_jobs(args: argparse.Namespace) -> None:
    jobs_dir = Path(JOBS_DIR)
    if not jobs_dir.exists():
        if getattr(args, "json", False):
            print("[]")
        else:
            print("No jobs.")
        return

    jobs = []
    for d in sorted(jobs_dir.iterdir()):
        if not d.is_dir():
            continue
        json_path = d / "job.json"
        try:
            with open(json_path) as fh:
                job = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        # Check if running worker is actually alive
        if job.get("status") == "running":
            pid = job.get("pid")
            if pid and not _pid_alive(pid):
                job["status"] = "dead"
        jobs.append(job)

    if not jobs:
        if getattr(args, "json", False):
            print("[]")
        else:
            print("No jobs.")
        return

    if getattr(args, "json", False):
        print(json.dumps(jobs, indent=2))
        return

    # Table output
    print(f"{'ID':<10} {'STATUS':<12} {'PROGRESS':<10} {'CREATED':<20} {'STEP'}")
    print("-" * 70)
    for job in jobs:
        short_id = job["id"][:8]
        status = job.get("status", "?")
        p = job.get("progress", {})
        pct = f"{p.get('percent', 0)}%"
        gen = f"{p.get('generation', 0)}/{p.get('total_generations', 0)}"
        progress = f"{pct} ({gen})"
        created = job.get("created_at", "?")[:19]
        step = f"{p.get('step', 0)}/{p.get('total_steps', 0)}"
        print(f"{short_id:<10} {status:<12} {progress:<10} {created:<20} {step}")


def cmd_get_job(args: argparse.Namespace) -> None:
    job = load_job(args.id)
    if job.get("status") == "running":
        pid = job.get("pid")
        if pid and not _pid_alive(pid):
            job["status"] = "dead"
    print(json.dumps(job, indent=2))


def cmd_cancel_job(args: argparse.Namespace) -> None:
    job = load_job(args.id)
    status = job.get("status")
    if status not in ("queued", "running"):
        die(f"Cannot cancel job with status '{status}'")

    pid = job.get("pid")
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    job["status"] = "cancelled"
    job["completed_at"] = datetime.now().isoformat()
    save_job(job)
    print(f"Cancelled job {job['id'][:8]}")


def cmd_get_job_log(args: argparse.Namespace) -> None:
    resolved = _resolve_job_id(args.id)
    log_path = _job_log_path(resolved)

    n = args.n or 20

    if not args.follow:
        # Non-follow: show last N lines, exit silently if no log
        if not log_path.exists() or log_path.stat().st_size == 0:
            return
        for line in _tail(log_path, n):
            sys.stdout.write(line)
        sys.stdout.flush()
        return

    # Follow mode: wait for log file to appear
    try:
        while not log_path.exists() or log_path.stat().st_size == 0:
            time.sleep(0.5)

        for line in _tail(log_path, n):
            sys.stdout.write(line)
        sys.stdout.flush()

        with open(log_path) as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue
                time.sleep(0.5)
    except (BrokenPipeError, KeyboardInterrupt):
        pass


def cmd_log(args: argparse.Namespace) -> None:
    log_path = Path(LOG_DIR) / "tts.log"
    n = args.n or 20

    if not args.follow:
        # Non-follow: show last N lines, exit silently if no logs
        if not log_path.exists() or log_path.stat().st_size == 0:
            return
        for line in _tail(log_path, n):
            sys.stdout.write(line)
        sys.stdout.flush()
        return

    # Follow mode: wait for log file to appear
    try:
        while not log_path.exists() or log_path.stat().st_size == 0:
            time.sleep(0.5)

        for line in _tail(log_path, n):
            sys.stdout.write(line)
        sys.stdout.flush()

        with open(log_path) as f:
            f.seek(0, 2)
            while True:
                where = f.tell()
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue
                # Detect truncation (day rotation)
                try:
                    size = os.path.getsize(log_path)
                except OSError:
                    size = 0
                if size < where:
                    f.seek(0)
                time.sleep(0.5)
    except (BrokenPipeError, KeyboardInterrupt):
        pass


def cmd_tokenize(args: argparse.Namespace) -> None:
    from qwen_tts import Qwen3TTSTokenizer

    log_mgr = setup_logging()
    try:
        tok_path = resolve_tokenizer_path(args.models_dir, args.tokenizer)
        print(f"Loading tokenizer: {tok_path} ...")
        tokenizer = Qwen3TTSTokenizer.from_pretrained(tok_path, device_map=DEVICE)
        print("Tokenizer loaded.")

        audio_path = safe_path(args.audio)
        print(f"Encoding: {audio_path} ...")
        encoded = tokenizer.encode(audio_path)

        for i, codes in enumerate(encoded.audio_codes):
            print(f"  Segment {i}: codes shape {tuple(codes.shape)}")

        print("Decoding back to waveform ...")
        wavs, sr = tokenizer.decode(encoded)
        save_audio(wavs, sr, args.output)
    finally:
        log_mgr.close()
