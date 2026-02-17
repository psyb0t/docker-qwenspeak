import os
import sys
import time
from datetime import datetime
from pathlib import Path

from tts.constants import LOG_DIR, LOG_RETENTION_DEFAULT


def parse_duration(s: str) -> float:
    """Parse duration string like '7d', '1w', '24h' to seconds."""
    s = s.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if s and s[-1] in units:
        return float(s[:-1]) * units[s[-1]]
    return float(s) * 86400


class LogManager:
    """Dual-file logging with day rotation and cleanup."""

    def __init__(self) -> None:
        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = self.log_dir / "tts.log"
        self._today: str | None = None
        self._daily_fh = None
        self._current_fh = None
        self._job_fh = None
        self._open_files()
        self._cleanup()

    def _open_files(self) -> None:
        today = datetime.now().strftime("%Y_%m_%d")
        daily_path = self.log_dir / f"{today}_tts.log"

        if self._today == today:
            return

        if self._daily_fh:
            self._daily_fh.close()
        if self._current_fh:
            self._current_fh.close()

        # New day and no daily file yet = truncate current log
        if not daily_path.exists():
            with open(self.current_log, "w"):
                pass

        self._today = today
        self._daily_fh = open(daily_path, "a")
        self._current_fh = open(self.current_log, "a")

    def write(self, data: str) -> None:
        if not data:
            return
        self._open_files()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for line in data.splitlines(True):
            if line.strip():
                stamped = f"[{ts}] {line}"
            else:
                stamped = line
            self._daily_fh.write(stamped)
            self._current_fh.write(stamped)
            if self._job_fh:
                self._job_fh.write(stamped)
        self._daily_fh.flush()
        self._current_fh.flush()
        if self._job_fh:
            self._job_fh.flush()

    def set_job_log(self, path: str) -> None:
        self.clear_job_log()
        self._job_fh = open(path, "a")

    def clear_job_log(self) -> None:
        if self._job_fh:
            self._job_fh.close()
            self._job_fh = None

    def _cleanup(self) -> None:
        retention_str = os.environ.get("TTS_LOG_RETENTION", LOG_RETENTION_DEFAULT)
        try:
            max_age = parse_duration(retention_str)
        except (ValueError, IndexError):
            max_age = parse_duration(LOG_RETENTION_DEFAULT)
        now = time.time()
        for f in self.log_dir.glob("*_*_*_tts.log"):
            try:
                if now - f.stat().st_mtime > max_age:
                    f.unlink()
            except OSError:
                pass

    def close(self) -> None:
        if self._daily_fh:
            self._daily_fh.close()
        if self._current_fh:
            self._current_fh.close()
        self.clear_job_log()


class TeeWriter:
    """Tee stdout/stderr to LogManager with timestamps."""

    def __init__(self, original, log_manager: LogManager):
        self.original = original
        self.log_manager = log_manager

    def write(self, data):
        if data:
            self.original.write(data)
            self.log_manager.write(data)

    def flush(self):
        self.original.flush()

    def fileno(self):
        return self.original.fileno()

    def isatty(self):
        return False


def setup_logging() -> LogManager:
    """Install TeeWriters on stdout/stderr. Returns LogManager for cleanup."""
    mgr = LogManager()
    sys.stdout = TeeWriter(sys.stdout, mgr)
    sys.stderr = TeeWriter(sys.stderr, mgr)
    return mgr
