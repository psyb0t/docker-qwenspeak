#!/usr/bin/env python3
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
  tts cancel-job <id>    Cancel a running job
  tts log [-f] [-n N]    View TTS logs
  tts tokenize <audio>   Encode/decode audio through the speech tokenizer

Examples:
  # Submit a job
  ssh tts@host "tts" < job.yaml
  # {"id": "...", "status": "pending", ...}

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
import gc
import json
import os
import shutil
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile
import torch
import yaml
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODELS_DIR = "/models"
LOG_DIR = "/var/log/tts"
LOG_RETENTION_DEFAULT = "7d"
JOBS_DIR = "/jobs"

MODELS = {
    "custom-voice-1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "custom-voice-0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "voice-design-1.7b": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base-1.7b": "Qwen3-TTS-12Hz-1.7B-Base",
    "base-0.6b": "Qwen3-TTS-12Hz-0.6B-Base",
}

TOKENIZERS = {
    "12hz": "Qwen3-TTS-Tokenizer-12Hz",
}

SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]

LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

VALID_MODES = ["custom-voice", "voice-design", "voice-clone"]

GENERATION_DEFAULTS = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
    "streaming": False,
    "no_sample": False,
}

DEVICE = os.environ.get("PROCESSING_UNIT", "cpu")

GLOBAL_DEFAULTS = {
    "dtype": "float32",
    "models_dir": DEFAULT_MODELS_DIR,
    "flash_attn": False,
    **GENERATION_DEFAULTS,
}

SPEAKER_INFO = {
    "Vivian": ("Female", "Chinese", "Bright, slightly edgy young voice"),
    "Serena": ("Female", "Chinese", "Warm, gentle young voice"),
    "Uncle_Fu": ("Male", "Chinese", "Seasoned, low mellow timbre"),
    "Dylan": ("Male", "Chinese", "Youthful Beijing dialect, clear natural timbre"),
    "Eric": ("Male", "Chinese", "Lively Chengdu/Sichuan dialect, slightly husky"),
    "Ryan": ("Male", "English", "Dynamic with strong rhythmic drive"),
    "Aiden": ("Male", "English", "Sunny American, clear midrange"),
    "Ono_Anna": ("Female", "Japanese", "Playful, light nimble timbre"),
    "Sohee": ("Female", "Korean", "Warm with rich emotion"),
}

YAML_TEMPLATE = """\
# Qwen3-TTS YAML config
# Pipe this to tts via stdin: ssh tts@host "tts" < job.yaml
# Returns a job UUID immediately. Poll with: ssh tts@host "tts get-job <id>"

# Global settings (apply to all steps unless overridden)
dtype: float32             # float32, float16, bfloat16 (float16/bfloat16 GPU only)
models_dir: /models
flash_attn: false          # FlashAttention-2 (GPU only)

# Generation defaults (override per-step or per-generation)
temperature: 0.9
top_k: 50
top_p: 1.0
repetition_penalty: 1.05
max_new_tokens: 2048
streaming: false
no_sample: false           # true = greedy decoding

steps:
  # --- custom-voice: preset speakers with optional emotion control ---
  - mode: custom-voice
    model_size: 1.7b       # 1.7b or 0.6b
    speaker: Ryan          # step default speaker
    language: English      # step default language (Auto, Chinese, English, Japanese, Korean, ...)
    generate:
      - text: "Hello world"
        output: hello.wav
      - text: "I cannot believe this!"
        speaker: Vivian    # override step speaker
        instruct: "Speak angrily"   # emotion/style (1.7B only)
        output: angry.wav

  # --- voice-design: describe the voice in natural language ---
  - mode: voice-design
    # model_size is always 1.7b for voice-design
    generate:
      - text: "Welcome to our store."
        instruct: "A warm, friendly young female voice with a cheerful tone"
        language: English
        output: welcome.wav

  # --- voice-clone: clone a voice from reference audio ---
  - mode: voice-clone
    model_size: 1.7b       # 1.7b or 0.6b
    ref_audio: /work/ref.wav        # step default (prompt reuse when shared)
    ref_text: "Transcript of ref"   # required unless x_vector_only
    language: Auto
    generate:
      - text: "First line in cloned voice"
        output: clone1.wav
      - text: "Second line"
        output: clone2.wav
      - text: "Different reference"
        ref_audio: /work/other.wav  # override ref for this one
        x_vector_only: true         # no transcript needed
        output: clone3.wav
"""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def parse_duration(s: str) -> float:
    """Parse duration string like '7d', '1w', '24h' to seconds."""
    s = s.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if s and s[-1] in units:
        return float(s[:-1]) * units[s[-1]]
    return float(s) * 86400


class LogManager:
    """Dual-file logging with day rotation and cleanup."""

    def __init__(self, job_log_path: str | None = None):
        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = self.log_dir / "tts.log"
        self._today = None
        self._daily_fh = None
        self._current_fh = None
        self._job_fh = None
        if job_log_path:
            self._job_fh = open(job_log_path, "a")
        self._open_files()
        self._cleanup()

    def _open_files(self):
        today = datetime.now().strftime("%Y_%m_%d")
        daily_path = self.log_dir / f"{today}_tts.log"

        if self._today == today:
            return

        # Close old handles
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

    def _cleanup(self):
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

    def close(self):
        if self._daily_fh:
            self._daily_fh.close()
        if self._current_fh:
            self._current_fh.close()
        if self._job_fh:
            self._job_fh.close()


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


def setup_logging(job_log_path: str | None = None) -> LogManager:
    """Install TeeWriters on stdout/stderr. Returns LogManager for cleanup."""
    mgr = LogManager(job_log_path=job_log_path)
    sys.stdout = TeeWriter(sys.stdout, mgr)
    sys.stderr = TeeWriter(sys.stderr, mgr)
    return mgr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        die(f"Unknown dtype '{name}'. Choose from: {list(mapping)}")
    return mapping[name]


def resolve_model_path(models_dir: str, model_key: str) -> str:
    subdir = MODELS[model_key]
    path = Path(models_dir) / subdir
    if not path.is_dir():
        die(f"Model not found: {path}")
    return str(path)


def resolve_tokenizer_path(models_dir: str, variant: str) -> str:
    subdir = TOKENIZERS[variant]
    path = Path(models_dir) / subdir
    if not path.is_dir():
        die(f"Tokenizer not found: {path}")
    return str(path)


def load_model(
    model_path: str, device: str, dtype: torch.dtype, flash_attn: bool
) -> Qwen3TTSModel:
    kwargs = {"device_map": device, "dtype": dtype}
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    print(f"Loading model: {model_path} on {device} ({dtype}) ...")
    model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
    print("Model loaded.")
    return model


def unload_model(model: Qwen3TTSModel) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gen_kwargs(cfg: dict) -> dict:
    return {
        "do_sample": not cfg.get("no_sample", False),
        "temperature": cfg.get("temperature", 0.9),
        "top_k": cfg.get("top_k", 50),
        "top_p": cfg.get("top_p", 1.0),
        "repetition_penalty": cfg.get("repetition_penalty", 1.05),
        "max_new_tokens": cfg.get("max_new_tokens", 2048),
        "non_streaming_mode": not cfg.get("streaming", False),
    }


def save_audio(wavs: list[np.ndarray], sr: int, output: str) -> None:
    if len(wavs) == 1:
        soundfile.write(output, wavs[0], sr)
        print(f"Saved: {output}  ({len(wavs[0]) / sr:.2f}s, {sr} Hz)")
    else:
        stem = Path(output).stem
        suffix = Path(output).suffix or ".wav"
        parent = Path(output).parent
        for i, wav in enumerate(wavs):
            path = parent / f"{stem}_{i}{suffix}"
            soundfile.write(str(path), wav, sr)
            print(f"Saved: {path}  ({len(wav) / sr:.2f}s, {sr} Hz)")


def merge_config(*layers: dict) -> dict:
    """Merge config dicts, later layers override earlier ones. None values are skipped."""
    merged = {}
    for layer in layers:
        if layer:
            for k, v in layer.items():
                if v is not None:
                    merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------


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
    # Try prefix match on directory names
    matches = [
        d.name
        for d in Path(JOBS_DIR).iterdir()
        if d.is_dir() and d.name.startswith(job_id)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        die(f"Ambiguous job ID '{job_id}', matches: {[m[:8] for m in matches]}")
    die(f"Job not found: {job_id}")
    return ""  # unreachable, die exits


def load_job(job_id: str) -> dict:
    resolved = _resolve_job_id(job_id)
    path = _job_path(resolved)
    if not path.exists():
        die(f"Job not found: {job_id}")
    with open(path) as f:
        return json.load(f)


def delete_job(job: dict) -> None:
    d = _job_dir(job["id"])
    shutil.rmtree(d, ignore_errors=True)


def cleanup_stale_jobs() -> None:
    """Remove job directories whose worker PID is dead."""
    jobs_dir = Path(JOBS_DIR)
    if not jobs_dir.exists():
        return
    for d in jobs_dir.iterdir():
        if not d.is_dir():
            continue
        json_path = d / "job.json"
        try:
            with open(json_path) as fh:
                job = json.load(fh)
        except (json.JSONDecodeError, OSError):
            shutil.rmtree(d, ignore_errors=True)
            continue
        if job.get("status") != "running":
            continue
        pid = job.get("pid")
        if not pid or not _pid_alive(pid):
            shutil.rmtree(d, ignore_errors=True)


def create_job(config: dict) -> dict:
    """Create a new job from parsed YAML config."""
    steps = config.get("steps", [])
    total_gens = sum(len(s.get("generate", [])) for s in steps)
    job = {
        "id": str(uuid.uuid4()),
        "status": "pending",
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


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------


def gen_custom_voice(model: Qwen3TTSModel, cfg: dict) -> None:
    text = cfg.get("text")
    if not text:
        die("custom-voice generation missing 'text'")

    speaker = cfg.get("speaker", "Vivian")
    if speaker not in SPEAKERS:
        die(f"Unknown speaker '{speaker}'. Choose from: {SPEAKERS}")

    language = cfg.get("language", "Auto")
    instruct = cfg.get("instruct")
    output = cfg.get("output")
    if not output:
        die("custom-voice generation missing 'output'")

    model_size = cfg.get("model_size", "1.7b")
    if model_size == "0.6b" and instruct:
        print("Warning: 0.6B CustomVoice does not support instruct. Ignoring.")
        instruct = None

    kwargs = get_gen_kwargs(cfg)
    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct,
        **kwargs,
    )
    save_audio(wavs, sr, output)


def gen_voice_design(model: Qwen3TTSModel, cfg: dict) -> None:
    text = cfg.get("text")
    if not text:
        die("voice-design generation missing 'text'")

    instruct = cfg.get("instruct")
    if not instruct:
        die("voice-design generation missing 'instruct'")

    language = cfg.get("language", "Auto")
    output = cfg.get("output")
    if not output:
        die("voice-design generation missing 'output'")

    kwargs = get_gen_kwargs(cfg)
    wavs, sr = model.generate_voice_design(
        text=text,
        instruct=instruct,
        language=language,
        **kwargs,
    )
    save_audio(wavs, sr, output)


def gen_voice_clone(model: Qwen3TTSModel, cfg: dict, prompt=None) -> None:
    text = cfg.get("text")
    if not text:
        die("voice-clone generation missing 'text'")

    output = cfg.get("output")
    if not output:
        die("voice-clone generation missing 'output'")

    language = cfg.get("language", "Auto")
    kwargs = get_gen_kwargs(cfg)

    if prompt is not None:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            **kwargs,
        )
    else:
        ref_audio = cfg.get("ref_audio")
        if not ref_audio:
            die("voice-clone generation missing 'ref_audio'")

        x_vector_only = cfg.get("x_vector_only", False)
        ref_text = None if x_vector_only else cfg.get("ref_text")

        if not x_vector_only and not ref_text:
            die(
                "voice-clone generation missing 'ref_text' (required unless x_vector_only: true)"
            )

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            **kwargs,
        )
    save_audio(wavs, sr, output)


# ---------------------------------------------------------------------------
# Pipeline (async job runner)
# ---------------------------------------------------------------------------


def _run_generation(mode: str, model, cfg: dict, clone_prompt=None) -> None:
    """Dispatch a single generation to the right mode function."""
    if mode == "custom-voice":
        gen_custom_voice(model, cfg)
        return
    if mode == "voice-design":
        gen_voice_design(model, cfg)
        return
    # voice-clone
    use_prompt = clone_prompt if (clone_prompt and not cfg.get("ref_audio")) else None
    gen_voice_clone(model, cfg, prompt=use_prompt)


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
    flash_attn = merged.get("flash_attn", False)
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
                    ref_audio=step_cfg["ref_audio"],
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                )

        for i, gen in enumerate(generations):
            cfg = merge_config(globals_cfg, step_cfg, gen)
            print(
                f"[{mode}] Generating {i + 1}/{len(generations)}: {cfg.get('text', '')[:60]}..."
            )

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
    """Run the full TTS pipeline as a background job."""
    cleanup_stale_jobs()

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
        delete_job(job)

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        save_job(job)
        print(f"\nFailed: {e}")
        delete_job(job)


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


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


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
    # Check if running worker is actually alive
    if job.get("status") == "running":
        pid = job.get("pid")
        if pid and not _pid_alive(pid):
            job["status"] = "dead"
    print(json.dumps(job, indent=2))


def cmd_cancel_job(args: argparse.Namespace) -> None:
    job = load_job(args.id)
    if job.get("status") not in ("pending", "running"):
        die(f"Cannot cancel job with status '{job.get('status')}'")

    pid = job.get("pid")
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    job["status"] = "cancelled"
    job["completed_at"] = datetime.now().isoformat()
    save_job(job)
    delete_job(job)
    print(f"Cancelled job {job['id'][:8]}")


def cmd_get_job_log(args: argparse.Namespace) -> None:
    resolved = _resolve_job_id(args.id)
    log_path = _job_log_path(resolved)

    if not log_path.exists() or log_path.stat().st_size == 0:
        print("No log output yet.")
        return

    n = args.n or 20

    lines = _tail(log_path, n)
    for line in lines:
        sys.stdout.write(line)
    sys.stdout.flush()

    if not args.follow:
        return

    try:
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


def cmd_tokenize(args: argparse.Namespace) -> None:
    log_mgr = setup_logging()
    try:
        tok_path = resolve_tokenizer_path(args.models_dir, args.tokenizer)
        print(f"Loading tokenizer: {tok_path} ...")
        tokenizer = Qwen3TTSTokenizer.from_pretrained(tok_path, device_map=DEVICE)
        print("Tokenizer loaded.")

        print(f"Encoding: {args.audio} ...")
        encoded = tokenizer.encode(args.audio)

        for i, codes in enumerate(encoded.audio_codes):
            print(f"  Segment {i}: codes shape {tuple(codes.shape)}")

        print("Decoding back to waveform ...")
        wavs, sr = tokenizer.decode(encoded)
        save_audio(wavs, sr, args.output)
    finally:
        log_mgr.close()


def _tail(path: Path, n: int) -> list[str]:
    """Read last n lines from a file."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    return lines[-n:] if len(lines) > n else lines


def cmd_log(args: argparse.Namespace) -> None:
    log_path = Path(LOG_DIR) / "tts.log"

    if not log_path.exists() or log_path.stat().st_size == 0:
        print("No logs yet.")
        return

    n = args.n or 20

    lines = _tail(log_path, n)
    for line in lines:
        sys.stdout.write(line)
    sys.stdout.flush()

    if not args.follow:
        return

    # Follow mode: tail -f with truncation detection
    try:
        with open(log_path) as f:
            f.seek(0, 2)
            while True:
                where = f.tell()
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
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


# ---------------------------------------------------------------------------
# YAML validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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

    p_cj = sub.add_parser("cancel-job", help="Cancel a running job")
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
    p_gjl.add_argument(
        "-n", type=int, default=20, help="Number of lines to show (default: 20)"
    )
    p_gjl.set_defaults(func=cmd_get_job_log)

    p_log = sub.add_parser("log", help="View TTS logs")
    p_log.add_argument(
        "-f",
        "--follow",
        action="store_true",
        default=False,
        help="Follow log output (like tail -f)",
    )
    p_log.add_argument(
        "-n", type=int, default=20, help="Number of lines to show (default: 20)"
    )
    p_log.set_defaults(func=cmd_log)

    p_tok = sub.add_parser(
        "tokenize", help="Encode audio to speech tokens and decode back"
    )
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

    # Child: become daemon
    daemonize()

    log_mgr = setup_logging(job_log_path=str(_job_log_path(job["id"])))
    try:
        run_pipeline(job)
    finally:
        log_mgr.close()


if __name__ == "__main__":
    main()
