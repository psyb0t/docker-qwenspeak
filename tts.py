#!/usr/bin/env python3
"""Qwen3-TTS: Full-featured text-to-speech with all models and options.

Supports three modes:
  custom-voice  - Use one of 9 predefined premium speakers with optional emotion control
  voice-design  - Describe the voice you want in natural language
  voice-clone   - Clone a voice from a reference audio sample

Models are loaded from a local directory (default: /models), expected layout:
  /models/
    Qwen3-TTS-12Hz-1.7B-CustomVoice/
    Qwen3-TTS-12Hz-0.6B-CustomVoice/
    Qwen3-TTS-12Hz-1.7B-VoiceDesign/
    Qwen3-TTS-12Hz-1.7B-Base/
    Qwen3-TTS-12Hz-0.6B-Base/
    Qwen3-TTS-Tokenizer-12Hz/

Download models:
    pip install -U "huggingface_hub[cli]"
    huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen3-TTS-12Hz-0.6B-Base

Examples:
  # CustomVoice - predefined speakers (CPU)
  python main.py custom-voice "Hello, how are you?" --speaker Vivian --language English

  # CustomVoice - with emotion instruction (1.7B only)
  python main.py custom-voice "I can't believe this happened!" --speaker Ryan --instruct "Speak angrily"

  # CustomVoice - GPU with bfloat16
  python main.py custom-voice "Hello!" --speaker Aiden --device cuda:0 --dtype bfloat16

  # VoiceDesign - describe the voice in natural language
  python main.py voice-design "Welcome to our store." \
      --instruct "A warm, friendly young female voice with a cheerful tone"

  # Voice cloning - clone from reference audio (ICL mode)
  python main.py voice-clone "This is my cloned voice speaking." \
      --ref-audio reference.wav --ref-text "Original text from the reference audio."

  # Voice cloning - x-vector only mode (no transcript needed)
  python main.py voice-clone "This is my cloned voice." \
      --ref-audio reference.wav --x-vector-only

  # Custom models directory
  python main.py -m /mnt/hdd/models/qwen3-tts custom-voice "Hi!" --speaker Ryan

  # List available speakers
  python main.py list-speakers

  # Encode/decode audio through the speech tokenizer
  python main.py tokenize input.wav --output reconstructed.wav
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile
import torch

from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODELS_DIR = "/models"

MODELS = {
    # CustomVoice models (predefined speakers)
    "custom-voice-1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "custom-voice-0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    # VoiceDesign model (natural language voice description)
    "voice-design-1.7b": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    # Base models (voice cloning)
    "base-1.7b": "Qwen3-TTS-12Hz-1.7B-Base",
    "base-0.6b": "Qwen3-TTS-12Hz-0.6B-Base",
}

TOKENIZERS = {
    "12hz": "Qwen3-TTS-Tokenizer-12Hz",
}

SPEAKERS = [
    "Vivian",     # Bright young female – Chinese
    "Serena",     # Warm gentle female – Chinese
    "Uncle_Fu",   # Seasoned low male – Chinese
    "Dylan",      # Youthful Beijing male – Chinese
    "Eric",       # Lively Chengdu male – Chinese (Sichuan)
    "Ryan",       # Dynamic male – English
    "Aiden",      # Sunny American male – English
    "Ono_Anna",   # Playful female – Japanese
    "Sohee",      # Warm female – Korean
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

# Default generation parameters matching upstream defaults
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 1.0
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_NEW_TOKENS = 2048


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        raise ValueError(f"Unknown dtype '{name}'. Choose from: {list(mapping)}")
    return mapping[name]


def resolve_model_path(models_dir: str, model_key: str) -> str:
    """Resolve a model key (e.g. 'custom-voice-1.7b') to a local path."""
    subdir = MODELS[model_key]
    path = Path(models_dir) / subdir
    if not path.is_dir():
        raise FileNotFoundError(f"Model not found: {path}")
    return str(path)


def resolve_tokenizer_path(models_dir: str, variant: str) -> str:
    """Resolve a tokenizer variant (e.g. '12hz') to a local path."""
    subdir = TOKENIZERS[variant]
    path = Path(models_dir) / subdir
    if not path.is_dir():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    return str(path)


def load_model(model_path: str, device: str, dtype: torch.dtype, flash_attn: bool) -> Qwen3TTSModel:
    kwargs = {"device_map": device, "dtype": dtype}
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    print(f"Loading model: {model_path} on {device} ({dtype}) ...")
    model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
    print("Model loaded.")
    return model


def generation_kwargs(args: argparse.Namespace) -> dict:
    """Extract sampling / generation kwargs from parsed args."""
    return {
        "do_sample": not args.no_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
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


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_custom_voice(args: argparse.Namespace) -> None:
    """Generate speech using a predefined speaker voice."""
    model_key = f"custom-voice-{args.model_size}"
    model_path = args.checkpoint or resolve_model_path(args.models_dir, model_key)
    model = load_model(model_path, args.device, resolve_dtype(args.dtype), args.flash_attn)

    if args.model_size == "0.6b" and args.instruct:
        print("Warning: 0.6B CustomVoice model does not support --instruct. Ignoring.")
        args.instruct = None

    kwargs = generation_kwargs(args)
    kwargs["non_streaming_mode"] = not args.streaming

    wavs, sr = model.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        instruct=args.instruct or None,
        **kwargs,
    )
    save_audio(wavs, sr, args.output)


def cmd_voice_design(args: argparse.Namespace) -> None:
    """Generate speech using a natural language voice description."""
    model_path = args.checkpoint or resolve_model_path(args.models_dir, "voice-design-1.7b")
    model = load_model(model_path, args.device, resolve_dtype(args.dtype), args.flash_attn)

    kwargs = generation_kwargs(args)
    kwargs["non_streaming_mode"] = not args.streaming

    wavs, sr = model.generate_voice_design(
        text=args.text,
        instruct=args.instruct,
        language=args.language,
        **kwargs,
    )
    save_audio(wavs, sr, args.output)


def cmd_voice_clone(args: argparse.Namespace) -> None:
    """Clone a voice from reference audio."""
    model_key = f"base-{args.model_size}"
    model_path = args.checkpoint or resolve_model_path(args.models_dir, model_key)
    model = load_model(model_path, args.device, resolve_dtype(args.dtype), args.flash_attn)

    kwargs = generation_kwargs(args)
    kwargs["non_streaming_mode"] = not args.streaming

    if args.x_vector_only and args.ref_text:
        print("Note: --ref-text is ignored in x-vector-only mode.")

    # If creating a reusable prompt for multiple texts
    if args.prompt_reuse and isinstance(args.text, list) and len(args.text) > 1:
        print("Creating reusable voice clone prompt ...")
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=args.ref_audio,
            ref_text=None if args.x_vector_only else args.ref_text,
            x_vector_only_mode=args.x_vector_only,
        )
        all_wavs = []
        sr = None
        for i, text in enumerate(args.text):
            print(f"Generating {i + 1}/{len(args.text)} ...")
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=args.language,
                voice_clone_prompt=prompt_items,
                **kwargs,
            )
            all_wavs.extend(wavs)
        save_audio(all_wavs, sr, args.output)
    else:
        text = args.text[0] if isinstance(args.text, list) and len(args.text) == 1 else args.text
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=None if args.x_vector_only else args.ref_text,
            x_vector_only_mode=args.x_vector_only,
            **kwargs,
        )
        save_audio(wavs, sr, args.output)


def cmd_list_speakers(args: argparse.Namespace) -> None:
    """List available predefined speakers."""
    print("Available speakers for CustomVoice models:")
    print()
    info = {
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
    for name, (gender, lang, desc) in info.items():
        print(f"  {name:<12} {gender:<8} {lang:<10} {desc}")


def cmd_tokenize(args: argparse.Namespace) -> None:
    """Encode audio to speech tokens and decode back (round-trip)."""
    tok_path = args.checkpoint or resolve_tokenizer_path(args.models_dir, args.tokenizer)
    print(f"Loading tokenizer: {tok_path} ...")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(tok_path, device_map=args.device)
    print("Tokenizer loaded.")

    print(f"Encoding: {args.audio} ...")
    encoded = tokenizer.encode(args.audio)

    for i, codes in enumerate(encoded.audio_codes):
        print(f"  Segment {i}: codes shape {tuple(codes.shape)}")

    print("Decoding back to waveform ...")
    wavs, sr = tokenizer.decode(encoded)
    save_audio(wavs, sr, args.output)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS: text-to-speech with all models and options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models-dir", "-m", default=DEFAULT_MODELS_DIR, help=f"Base directory containing model subdirs (default: {DEFAULT_MODELS_DIR})")

    # ---- shared args ----
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--device", default="cpu", help="Device (default: cpu)")
    shared.add_argument("--dtype", default="float32", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"], help="Model dtype (default: float32)")
    shared.add_argument("--flash-attn", action="store_true", default=False, help="Use FlashAttention-2 (GPU only)")
    shared.add_argument("--output", "-o", default="output.wav", help="Output file path (default: output.wav)")
    shared.add_argument("--checkpoint", "-c", default=None, help="Override model path (absolute path, bypasses --models-dir)")

    # ---- generation args ----
    gen = argparse.ArgumentParser(add_help=False)
    gen.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    gen.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Top-k sampling (default: {DEFAULT_TOP_K})")
    gen.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help=f"Top-p / nucleus sampling (default: {DEFAULT_TOP_P})")
    gen.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY, help=f"Repetition penalty (default: {DEFAULT_REPETITION_PENALTY})")
    gen.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help=f"Max codec tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})")
    gen.add_argument("--no-sample", action="store_true", default=False, help="Disable sampling (greedy decoding)")
    gen.add_argument("--streaming", action="store_true", default=False, help="Use streaming mode (lower latency, simulates char-by-char input)")

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- custom-voice ----
    p_cv = sub.add_parser("custom-voice", parents=[shared, gen], help="Predefined speaker voices with optional emotion control")
    p_cv.add_argument("text", help="Text to synthesize")
    p_cv.add_argument("--speaker", "-s", default="Vivian", choices=SPEAKERS, help="Speaker name (default: Vivian)")
    p_cv.add_argument("--language", "-l", default="Auto", choices=LANGUAGES, help="Language (default: Auto)")
    p_cv.add_argument("--instruct", "-i", default=None, help="Emotion/style instruction, e.g. 'Speak angrily' (1.7B only)")
    p_cv.add_argument("--model-size", default="1.7b", choices=["1.7b", "0.6b"], help="Model size (default: 1.7b)")
    p_cv.set_defaults(func=cmd_custom_voice)

    # ---- voice-design ----
    p_vd = sub.add_parser("voice-design", parents=[shared, gen], help="Describe the desired voice in natural language")
    p_vd.add_argument("text", help="Text to synthesize")
    p_vd.add_argument("--instruct", "-i", required=True, help="Natural language voice description, e.g. 'A warm young female voice with a cheerful tone'")
    p_vd.add_argument("--language", "-l", default="Auto", choices=LANGUAGES, help="Language (default: Auto)")
    p_vd.set_defaults(func=cmd_voice_design)

    # ---- voice-clone ----
    p_vc = sub.add_parser("voice-clone", parents=[shared, gen], help="Clone a voice from reference audio")
    p_vc.add_argument("text", nargs="+", help="Text(s) to synthesize (multiple texts reuse the same voice prompt)")
    p_vc.add_argument("--ref-audio", "-r", required=True, help="Reference audio (file path, URL, or base64)")
    p_vc.add_argument("--ref-text", "-t", default=None, help="Transcript of reference audio (required for ICL mode, ignored with --x-vector-only)")
    p_vc.add_argument("--x-vector-only", action="store_true", default=False, help="Use only speaker embedding, no reference transcript needed")
    p_vc.add_argument("--language", "-l", default="Auto", choices=LANGUAGES, help="Language (default: Auto)")
    p_vc.add_argument("--model-size", default="1.7b", choices=["1.7b", "0.6b"], help="Model size (default: 1.7b)")
    p_vc.add_argument("--prompt-reuse", action="store_true", default=False, help="Pre-compute voice prompt and reuse across multiple texts (more efficient)")
    p_vc.set_defaults(func=cmd_voice_clone)

    # ---- list-speakers ----
    p_ls = sub.add_parser("list-speakers", help="List available predefined speakers")
    p_ls.set_defaults(func=cmd_list_speakers)

    # ---- tokenize ----
    p_tok = sub.add_parser("tokenize", parents=[shared], help="Encode audio to speech tokens and decode back (round-trip)")
    p_tok.add_argument("audio", help="Input audio file path")
    p_tok.add_argument("--tokenizer", default="12hz", choices=["12hz"], help="Tokenizer variant (default: 12hz)")
    p_tok.set_defaults(func=cmd_tokenize)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate voice-clone requires either --ref-text or --x-vector-only
    if args.command == "voice-clone" and not args.x_vector_only and not args.ref_text:
        parser.error("voice-clone requires --ref-text (transcript of reference audio) unless --x-vector-only is set")

    args.func(args)


if __name__ == "__main__":
    main()
