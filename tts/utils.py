from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile

from tts.constants import MODELS, TOKENIZERS

if TYPE_CHECKING:
    import torch
    from qwen_tts import Qwen3TTSModel


WORK_DIR = Path("/work")


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def safe_path(p: str) -> str:
    """Resolve user path under /work. Absolute paths get remapped."""
    resolved = (WORK_DIR / p.lstrip("/")).resolve()
    if not str(resolved).startswith(str(WORK_DIR.resolve())):
        die(f"Path traversal blocked: {p}")
    return str(resolved)


def merge_config(*layers: dict) -> dict:
    """Merge config dicts, later layers override earlier ones. None values skipped."""
    merged: dict = {}
    for layer in layers:
        if layer:
            for k, v in layer.items():
                if v is not None:
                    merged[k] = v
    return merged


def _tail(path: Path, n: int) -> list[str]:
    """Read last n lines from a file."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    return lines[-n:] if len(lines) > n else lines


def resolve_dtype(name: str) -> torch.dtype:
    import torch

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


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def load_model(model_path: str, device: str, dtype: torch.dtype, flash_attn: bool) -> Qwen3TTSModel:
    import torch
    from qwen_tts import Qwen3TTSModel

    if flash_attn and dtype == torch.float32:
        print("flash_attn requires fp16/bf16 — auto-switching to bfloat16")
        dtype = torch.bfloat16
    kwargs: dict = {"device_map": device, "dtype": dtype}
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    print(f"Loading model: {model_path} on {device} ({dtype}) ...")
    model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
    print("Model loaded.")
    return model


def unload_model(model: Qwen3TTSModel) -> None:
    import torch

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
    output = safe_path(output)
    if len(wavs) == 1:
        soundfile.write(output, wavs[0], sr)
        print(f"Saved: {output}  ({len(wavs[0]) / sr:.2f}s, {sr} Hz)")
        return
    stem = Path(output).stem
    suffix = Path(output).suffix or ".wav"
    parent = Path(output).parent
    for i, wav in enumerate(wavs):
        path = parent / f"{stem}_{i}{suffix}"
        soundfile.write(str(path), wav, sr)
        print(f"Saved: {path}  ({len(wav) / sr:.2f}s, {sr} Hz)")
