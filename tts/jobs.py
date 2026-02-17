import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

from tts.constants import JOBS_DIR, MAX_QUEUE_DEFAULT
from tts.utils import die

ONE_DAY = 86400
ONE_WEEK = 604800


def _job_dir(job_id: str) -> Path:
    return Path(JOBS_DIR) / job_id


def _job_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _job_log_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.log"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def save_job(job: dict) -> None:
    """Atomic write: tmp file then rename."""
    path = _job_path(job["id"])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(job, f, indent=2)
    os.rename(str(tmp), str(path))


def _resolve_job_id(job_id: str) -> str:
    """Resolve a full or prefix job ID to the actual job UUID."""
    if _job_dir(job_id).is_dir():
        return job_id
    matches = [d.name for d in Path(JOBS_DIR).iterdir() if d.is_dir() and d.name.startswith(job_id)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        die(f"Ambiguous job ID '{job_id}', matches: {[m[:8] for m in matches]}")
    die(f"Job not found: {job_id}")
    return ""  # unreachable


def load_job(job_id: str) -> dict:
    resolved = _resolve_job_id(job_id)
    path = _job_path(resolved)
    if not path.exists():
        die(f"Job not found: {job_id}")
    with open(path) as f:
        return json.load(f)


def load_job_raw(job_id: str) -> dict | None:
    """Load job by exact ID without dying. Returns None if not found."""
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def delete_job(job: dict) -> None:
    d = _job_dir(job["id"])
    shutil.rmtree(d, ignore_errors=True)


def create_job(config: dict) -> dict:
    """Create a new job from parsed YAML config."""
    steps = config.get("steps", [])
    total_gens = sum(len(s.get("generate", [])) for s in steps)
    job = {
        "id": str(uuid.uuid4()),
        "status": "queued",
        "pid": None,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "progress": {
            "step": 0,
            "total_steps": len(steps),
            "generation": 0,
            "total_generations": total_gens,
            "percent": 0,
        },
        "outputs": [],
        "config": config,
    }
    os.makedirs(_job_dir(job["id"]), exist_ok=True)
    save_job(job)
    return job


def count_active_jobs() -> int:
    """Count jobs with status queued or running."""
    jobs_dir = Path(JOBS_DIR)
    if not jobs_dir.exists():
        return 0
    count = 0
    for d in jobs_dir.iterdir():
        if not d.is_dir():
            continue
        json_path = d / "job.json"
        try:
            with open(json_path) as fh:
                job = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if job.get("status") in ("queued", "running"):
            count += 1
    return count


def max_queue() -> int:
    try:
        return int(os.environ.get("TTS_MAX_QUEUE", MAX_QUEUE_DEFAULT))
    except (ValueError, TypeError):
        return MAX_QUEUE_DEFAULT


def pick_next_queued_job() -> dict | None:
    """Find the oldest queued job by created_at. Returns None if none."""
    jobs_dir = Path(JOBS_DIR)
    if not jobs_dir.exists():
        return None
    best = None
    best_time = None
    for d in jobs_dir.iterdir():
        if not d.is_dir():
            continue
        json_path = d / "job.json"
        try:
            with open(json_path) as fh:
                job = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if job.get("status") != "queued":
            continue
        created = job.get("created_at", "")
        if best_time is None or created < best_time:
            best = job
            best_time = created
    return best


def cleanup_jobs() -> None:
    """Time-based job cleanup and stale PID detection."""
    jobs_dir = Path(JOBS_DIR)
    if not jobs_dir.exists():
        return
    now = time.time()
    for d in jobs_dir.iterdir():
        if not d.is_dir():
            continue
        json_path = d / "job.json"
        try:
            mtime = json_path.stat().st_mtime
            with open(json_path) as fh:
                job = json.load(fh)
        except (OSError, json.JSONDecodeError):
            shutil.rmtree(d, ignore_errors=True)
            continue

        age = now - mtime
        status = job.get("status")

        # Zombie: anything older than 1 week
        if age > ONE_WEEK:
            shutil.rmtree(d, ignore_errors=True)
            continue

        # Completed jobs older than 1 day
        if status == "completed" and age > ONE_DAY:
            shutil.rmtree(d, ignore_errors=True)
            continue

        # Running but PID dead → mark failed
        if status == "running":
            pid = job.get("pid")
            if pid and not _pid_alive(pid):
                job["status"] = "failed"
                job["error"] = "Worker process died"
                job["completed_at"] = datetime.now().isoformat()
                save_job(job)
