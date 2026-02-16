#!/usr/bin/env python3
"""Qwen3-TTS: YAML-driven text-to-speech pipeline.

Pipe a YAML config via stdin to run multi-step TTS generation.
Each step loads a model, runs all generations, then unloads it.

Usage:
  tts                    Read YAML from stdin, run pipeline
  tts print-yaml         Print a template YAML config to stdout
  tts list-speakers      List available preset speakers
  tts log [-f] [-n N]    View TTS logs
  tts tokenize <audio>   Encode/decode audio through the speech tokenizer

Examples:
  # Dump template, edit it, run it
  ssh tts@host "tts print-yaml" > job.yaml
  vim job.yaml
  ssh tts@host "tts" < job.yaml

  # View logs
  ssh tts@host "tts log"
  ssh tts@host "tts log -f"
  ssh tts@host "tts log -n 100"

  # List speakers
  ssh tts@host "tts list-speakers"

  # Tokenize round-trip
  ssh tts@host "tts tokenize input.wav"
"""

import argparse
import gc
import os
import sys
import time
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
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
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

DEVICE = os.environ.get("TTS_DEVICE", "cpu")

GLOBAL_DEFAULTS = {
    "dtype": "float32",
    "models_dir": DEFAULT_MODELS_DIR,
    "flash_attn": False,
    **GENERATION_DEFAULTS,
}

SPEAKER_INFO = {
    "Vivian":   ("Female", "Chinese",  "Bright, slightly edgy young voice"),
    "Serena":   ("Female", "Chinese",  "Warm, gentle young voice"),
    "Uncle_Fu": ("Male",   "Chinese",  "Seasoned, low mellow timbre"),
    "Dylan":    ("Male",   "Chinese",  "Youthful Beijing dialect, clear natural timbre"),
    "Eric":     ("Male",   "Chinese",  "Lively Chengdu/Sichuan dialect, slightly husky"),
    "Ryan":     ("Male",   "English",  "Dynamic with strong rhythmic drive"),
    "Aiden":    ("Male",   "English",  "Sunny American, clear midrange"),
    "Ono_Anna": ("Female", "Japanese", "Playful, light nimble timbre"),
    "Sohee":    ("Female", "Korean",   "Warm with rich emotion"),
}

YAML_TEMPLATE = """\
# Qwen3-TTS YAML config
# Pipe this to tts via stdin: ssh tts@host "tts" < job.yaml

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

    def __init__(self):
        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = self.log_dir / "tts.log"
        self._today = None
        self._daily_fh = None
        self._current_fh = None
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
        self._daily_fh.flush()
        self._current_fh.flush()

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
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


def load_model(model_path: str, device: str, dtype: torch.dtype, flash_attn: bool) -> Qwen3TTSModel:
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
            die("voice-clone generation missing 'ref_text' (required unless x_vector_only: true)")

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
# Pipeline
# ---------------------------------------------------------------------------

def run_step(step: dict, globals_cfg: dict) -> None:
    mode = step.get("mode")
    if mode not in VALID_MODES:
        die(f"Invalid mode '{mode}'. Choose from: {VALID_MODES}")

    generations = step.get("generate")
    if not generations or not isinstance(generations, list):
        die(f"Step mode '{mode}' has no 'generate' list")

    # Merge global defaults into step config (without generate list)
    step_cfg = {k: v for k, v in step.items() if k != "generate"}

    # Resolve model
    model_size = merge_config(globals_cfg, step_cfg).get("model_size", "1.7b")
    device = DEVICE
    dtype = merge_config(globals_cfg, step_cfg).get("dtype", "float32")
    flash_attn = merge_config(globals_cfg, step_cfg).get("flash_attn", False)
    models_dir = merge_config(globals_cfg, step_cfg).get("models_dir", DEFAULT_MODELS_DIR)

    if mode == "custom-voice":
        model_key = f"custom-voice-{model_size}"
    elif mode == "voice-design":
        model_key = "voice-design-1.7b"
    elif mode == "voice-clone":
        model_key = f"base-{model_size}"
    else:
        die(f"Unknown mode: {mode}")

    model_path = resolve_model_path(models_dir, model_key)
    model = load_model(model_path, device, resolve_dtype(dtype), flash_attn)

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
            print(f"[{mode}] Generating {i + 1}/{len(generations)}: {cfg.get('text', '')[:60]}...")

            if mode == "custom-voice":
                gen_custom_voice(model, cfg)
            elif mode == "voice-design":
                gen_voice_design(model, cfg)
            elif mode == "voice-clone":
                use_prompt = clone_prompt if (clone_prompt and not gen.get("ref_audio")) else None
                gen_voice_clone(model, cfg, prompt=use_prompt)
    finally:
        print(f"Unloading model: {model_key}")
        unload_model(model)


def run_yaml(stream) -> None:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        die(f"Invalid YAML: {e}")

    if not isinstance(config, dict):
        die("YAML config must be a mapping")

    steps = config.get("steps")
    if not steps or not isinstance(steps, list):
        die("YAML config must have a 'steps' list")

    globals_cfg = {k: v for k, v in config.items() if k != "steps"}

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            die(f"Step {i + 1} must be a mapping")
        print(f"\n=== Step {i + 1}/{len(steps)}: {step.get('mode', '?')} ===")
        run_step(step, globals_cfg)

    print(f"\nDone. {len(steps)} step(s) completed.")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list_speakers() -> None:
    print("Available speakers for CustomVoice models:")
    print()
    for name, (gender, lang, desc) in SPEAKER_INFO.items():
        print(f"  {name:<12} {gender:<8} {lang:<10} {desc}")


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
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS: YAML-driven text-to-speech pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models-dir", "-m", default=DEFAULT_MODELS_DIR, help=f"Base directory for models (default: {DEFAULT_MODELS_DIR})")

    sub = parser.add_subparsers(dest="command")

    p_py = sub.add_parser("print-yaml", help="Print a template YAML config to stdout")
    p_py.set_defaults(func=lambda a: print(YAML_TEMPLATE))

    p_ls = sub.add_parser("list-speakers", help="List available preset speakers")
    p_ls.set_defaults(func=lambda a: cmd_list_speakers())

    p_log = sub.add_parser("log", help="View TTS logs")
    p_log.add_argument("-f", "--follow", action="store_true", default=False, help="Follow log output (like tail -f)")
    p_log.add_argument("-n", type=int, default=20, help="Number of lines to show (default: 20)")
    p_log.set_defaults(func=cmd_log)

    p_tok = sub.add_parser("tokenize", help="Encode audio to speech tokens and decode back")
    p_tok.add_argument("audio", help="Input audio file path")
    p_tok.add_argument("--output", "-o", default="output.wav", help="Output file path (default: output.wav)")
    p_tok.add_argument("--tokenizer", default="12hz", choices=["12hz"], help="Tokenizer variant (default: 12hz)")
    p_tok.set_defaults(func=cmd_tokenize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command:
        args.func(args)
        return

    # No subcommand → read YAML from stdin
    if sys.stdin.isatty():
        parser.print_help()
        sys.exit(1)

    log_mgr = setup_logging()
    try:
        run_yaml(sys.stdin)
    finally:
        log_mgr.close()


if __name__ == "__main__":
    main()
