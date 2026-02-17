from __future__ import annotations

from typing import TYPE_CHECKING

from tts.constants import SPEAKERS
from tts.utils import die, get_gen_kwargs, safe_path, save_audio

if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSModel


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
        ref_audio = safe_path(ref_audio)

        x_vector_only = cfg.get("x_vector_only", False)
        ref_text = None if x_vector_only else cfg.get("ref_text")

        if not x_vector_only and not ref_text:
            die("voice-clone generation missing 'ref_text'" " (required unless x_vector_only: true)")

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            **kwargs,
        )
    save_audio(wavs, sr, output)


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
