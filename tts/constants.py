import os

DEFAULT_MODELS_DIR = "/models"
LOG_DIR = "/var/log/tts"
LOG_RETENTION_DEFAULT = "7d"
JOBS_DIR = "/jobs"
MAX_QUEUE_DEFAULT = 50

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
flash_attn: auto           # auto-detects; set true/false to override

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
    ref_audio: ref.wav              # step default (prompt reuse when shared)
    ref_text: "Transcript of ref"   # required unless x_vector_only
    language: Auto
    generate:
      - text: "First line in cloned voice"
        output: clone1.wav
      - text: "Second line"
        output: clone2.wav
      - text: "Different reference"
        ref_audio: other.wav        # override ref for this one
        x_vector_only: true         # no transcript needed
        output: clone3.wav
"""
