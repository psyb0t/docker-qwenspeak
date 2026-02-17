from __future__ import annotations

import fcntl
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

from tts.constants import DEFAULT_MODELS_DIR, DEVICE, JOBS_DIR, VALID_MODES
from tts.generation import _run_generation
from tts.jobs import (
    _job_log_path,
    cleanup_jobs,
    load_job_raw,
    pick_next_queued_job,
    save_job,
)
from tts.logging import setup_logging
from tts.utils import (
    _has_flash_attn,
    load_model,
    merge_config,
    resolve_dtype,
    resolve_model_path,
    safe_path,
    unload_model,
)


class JobCancelled(Exception):
    pass


def run_step_tracked(step: dict, globals_cfg: dict, job: dict, gen_done: int) -> int:
    """Run a single step with job progress tracking. Returns updated gen_done."""
    mode = step.get("mode")
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {VALID_MODES}")

    generations = step.get("generate")
    if not generations or not isinstance(generations, list):
        raise ValueError(f"Step mode '{mode}' has no 'generate' list")

    step_cfg = {k: v for k, v in step.items() if k != "generate"}

    # Resolve model
    merged = merge_config(globals_cfg, step_cfg)
    model_size = merged.get("model_size", "1.7b")
    dtype = merged.get("dtype", "float32")
    flash_attn = merged.get("flash_attn", _has_flash_attn())
    models_dir = merged.get("models_dir", DEFAULT_MODELS_DIR)

    if mode == "custom-voice":
        model_key = f"custom-voice-{model_size}"
    elif mode == "voice-design":
        model_key = "voice-design-1.7b"
    else:
        model_key = f"base-{model_size}"

    model_path = resolve_model_path(models_dir, model_key)
    model = load_model(model_path, DEVICE, resolve_dtype(dtype), flash_attn)

    try:
        # For voice-clone with step-level ref_audio, create reusable prompt
        clone_prompt = None
        if mode == "voice-clone" and step_cfg.get("ref_audio"):
            shared_count = sum(1 for g in generations if not g.get("ref_audio"))
            if shared_count > 1:
                x_vector_only = step_cfg.get("x_vector_only", False)
                ref_text = None if x_vector_only else step_cfg.get("ref_text")
                print("Creating reusable voice clone prompt ...")
                clone_prompt = model.create_voice_clone_prompt(
                    ref_audio=safe_path(step_cfg["ref_audio"]),
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                )

        for i, gen in enumerate(generations):
            cfg = merge_config(globals_cfg, step_cfg, gen)
            print(f"[{mode}] Generating {i + 1}/{len(generations)}:" f" {cfg.get('text', '')[:60]}...")

            _run_generation(mode, model, cfg, clone_prompt)

            gen_done += 1
            total = job["progress"]["total_generations"]
            job["progress"]["generation"] = gen_done
            job["progress"]["percent"] = int(gen_done * 100 / total) if total else 100
            job["outputs"].append(cfg.get("output"))
            save_job(job)
    finally:
        print(f"Unloading model: {model_key}")
        unload_model(model)

    return gen_done


def run_pipeline(job: dict) -> None:
    """Run the full TTS pipeline for a single job."""

    def _on_sigterm(_signum, _frame):
        raise JobCancelled()

    prev_handler = signal.signal(signal.SIGTERM, _on_sigterm)

    job["status"] = "running"
    job["pid"] = os.getpid()
    job["started_at"] = datetime.now().isoformat()
    save_job(job)

    config = job["config"]
    steps = config.get("steps", [])
    globals_cfg = {k: v for k, v in config.items() if k != "steps"}
    gen_done = 0

    try:
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i + 1} must be a mapping")

            job["progress"]["step"] = i + 1
            save_job(job)

            print(f"\n=== Step {i + 1}/{len(steps)}: {step.get('mode', '?')} ===")
            gen_done = run_step_tracked(step, globals_cfg, job, gen_done)

        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        save_job(job)
        print(f"\nDone. {len(steps)} step(s) completed.")

    except JobCancelled:
        # Re-read from disk — cancel-job may have already set status
        disk_job = load_job_raw(job["id"])
        if disk_job and disk_job.get("status") == "cancelled":
            job.update(disk_job)
        else:
            job["status"] = "cancelled"
            job["completed_at"] = datetime.now().isoformat()
            save_job(job)
        print("\nJob cancelled.")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        save_job(job)
        print(f"\nFailed: {e}")

    finally:
        signal.signal(signal.SIGTERM, prev_handler)


def run_worker() -> None:
    """Worker loop: acquire lock, process queued jobs until none remain."""
    lock_path = Path(JOBS_DIR) / ".run.lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # Another worker is active, it will pick up our queued job
        lock_fd.close()
        return

    try:
        log_mgr = setup_logging()
        cleanup_jobs()
        while True:
            job = pick_next_queued_job()
            if not job:
                break
            log_mgr.set_job_log(str(_job_log_path(job["id"])))
            run_pipeline(job)
            log_mgr.clear_job_log()
        log_mgr.close()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def daemonize() -> None:
    """Detach from SSH session: setsid + redirect stdio to /dev/null."""
    os.setsid()
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    sys.stdin = open(os.devnull, "r")
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    # Ignore SIGHUP so SSH disconnect doesn't kill us
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
