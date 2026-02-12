# docker-qwenspeak

[Docker Hub](https://hub.docker.com/r/psyb0t/qwenspeak)

Qwen3-TTS text-to-speech over SSH. Pick a voice, clone a voice, design a voice - all through a single `tts` command. Models run locally, no API keys, no cloud bullshit.

Built on top of [psyb0t/lockbox](https://github.com/psyb0t/docker-lockbox) - see that repo for the security model, file operations, path sandboxing, and all the SSH lockdown details.

## Features

- **9 premium speakers** - male/female voices across Chinese, English, Japanese, Korean
- **Emotion/style control** - make any preset speaker happy, angry, sad, whatever
- **Voice design** - describe the voice you want in plain English and it generates it
- **Voice cloning** - clone any voice from a 3-second audio sample
- **10 languages** - Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **CPU** - runs on CPU out of the box, no GPU required

## Models

You need to download models locally before running. Pick what you need:

```bash
pip install -U "huggingface_hub[cli]"

# Required: speech tokenizer (used by all models)
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz

# CustomVoice: 9 preset speakers + emotion control
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice

# VoiceDesign: natural language voice descriptions
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign

# Base: voice cloning from reference audio
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen3-TTS-12Hz-0.6B-Base
```

Expected directory layout:

```
/your/models/dir/
  Qwen3-TTS-Tokenizer-12Hz/
  Qwen3-TTS-12Hz-1.7B-CustomVoice/
  Qwen3-TTS-12Hz-0.6B-CustomVoice/
  Qwen3-TTS-12Hz-1.7B-VoiceDesign/
  Qwen3-TTS-12Hz-1.7B-Base/
  Qwen3-TTS-12Hz-0.6B-Base/
```

## Quick Start

### docker run

```bash
docker pull psyb0t/qwenspeak

cat ~/.ssh/id_rsa.pub > authorized_keys
mkdir -p work host_keys

docker run -d \
  --name qwenspeak \
  --restart unless-stopped \
  --memory 4g \
  -p 2222:22 \
  -e "LOCKBOX_UID=$(id -u)" \
  -e "LOCKBOX_GID=$(id -g)" \
  -v $(pwd)/authorized_keys:/etc/lockbox/authorized_keys:ro \
  -v $(pwd)/host_keys:/etc/lockbox/host_keys \
  -v $(pwd)/work:/work \
  -v /path/to/your/models:/models:ro \
  psyb0t/qwenspeak

ssh -p 2222 tts@localhost "tts list-speakers"
```

### run.sh

```bash
cat ~/.ssh/id_rsa.pub > authorized_keys
MODELS_DIR=/path/to/your/models PORT=2222 ./run.sh
```

## Allowed Commands

| Command | Description |
|---------|-------------|
| `tts`   | Text-to-speech generation (the only command, that's it) |

## TTS Modes

### custom-voice - Preset Speakers

Pick from 9 built-in voices. The 1.7B model supports emotion/style control via `--instruct`.

```bash
# Basic
ssh tts@host "tts custom-voice 'Hello world' --speaker Ryan --language English"

# With emotion (1.7B only)
ssh tts@host "tts custom-voice 'I cannot believe this!' --speaker Vivian --instruct 'Speak angrily'"

# Smaller model
ssh tts@host "tts custom-voice 'Hello' --speaker Aiden --model-size 0.6b"
```

### voice-design - Describe the Voice

Tell it what voice you want in natural language. Only available as 1.7B.

```bash
ssh tts@host "tts voice-design 'Welcome to our store.' --instruct 'A warm, friendly young female voice with a cheerful tone'"

ssh tts@host "tts voice-design 'Breaking news today.' --instruct 'Deep authoritative male news anchor voice'"
```

### voice-clone - Clone Any Voice

Clone a voice from a reference audio file. Upload your reference first, then generate.

```bash
# Upload reference audio
ssh tts@host "put ref.wav" < my_voice.wav

# Clone with transcript (ICL mode - best quality)
ssh tts@host "tts voice-clone 'New text in my voice' --ref-audio /work/ref.wav --ref-text 'What I said in the ref clip'"

# Clone without transcript (x-vector only - no transcript needed)
ssh tts@host "tts voice-clone 'New text in my voice' --ref-audio /work/ref.wav --x-vector-only"

# Smaller model
ssh tts@host "tts voice-clone 'Hello' --ref-audio /work/ref.wav --x-vector-only --model-size 0.6b"
```

**Emotion trick for cloned voices:** record yourself with different emotions and use the matching reference file:

```bash
ssh tts@host "put refs/happy.wav" < me_happy.wav
ssh tts@host "put refs/angry.wav" < me_angry.wav

ssh tts@host "tts voice-clone 'Great news!' --ref-audio /work/refs/happy.wav --ref-text 'transcript' "
ssh tts@host "tts voice-clone 'This is unacceptable' --ref-audio /work/refs/angry.wav --ref-text 'transcript'"
```

## Available Speakers

| Speaker | Gender | Language | Description |
|---------|--------|----------|-------------|
| Vivian | Female | Chinese | Bright, slightly edgy young voice |
| Serena | Female | Chinese | Warm, gentle young voice |
| Uncle_Fu | Male | Chinese | Seasoned, low mellow timbre |
| Dylan | Male | Chinese | Youthful Beijing dialect, clear natural timbre |
| Eric | Male | Chinese | Lively Chengdu/Sichuan dialect, slightly husky |
| Ryan | Male | English | Dynamic with strong rhythmic drive |
| Aiden | Male | English | Sunny American, clear midrange |
| Ono_Anna | Female | Japanese | Playful, light nimble timbre |
| Sohee | Female | Korean | Warm with rich emotion |

## Generation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `output.wav` | Output file path (relative to /work) |
| `--language`, `-l` | `Auto` | Language: Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| `--model-size` | `1.7b` | Model size: 1.7b or 0.6b |
| `--device` | `cpu` | Device: cpu, cuda:0, etc. |
| `--dtype` | `float32` | Model dtype: float32, float16, bfloat16 (float16/bfloat16 GPU only) |
| `--flash-attn` | off | Use FlashAttention-2 (GPU only) |
| `--temperature` | `0.9` | Sampling temperature |
| `--top-k` | `50` | Top-k sampling |
| `--top-p` | `1.0` | Top-p / nucleus sampling |
| `--repetition-penalty` | `1.05` | Repetition penalty |
| `--max-new-tokens` | `2048` | Max codec tokens to generate |
| `--no-sample` | off | Greedy decoding |
| `--streaming` | off | Streaming mode (lower latency) |

## File Operations

All file paths are relative to /work. Traversal attempts get blocked, absolute paths get remapped under /work.

| Command | Description |
|---------|-------------|
| `put` | Upload file from stdin |
| `get` | Download file to stdout |
| `ls` | List /work or a subdirectory (`--json` for JSON output) |
| `rm` | Delete a file |
| `mkdir` | Create directory (recursive) |
| `rmdir` | Remove empty directory |
| `rrmdir` | Remove directory and everything in it recursively |

## Usage Examples

```bash
# Generate and immediately play
ssh tts@host "tts custom-voice 'Hello there' --speaker Ryan" && \
ssh tts@host "get output.wav" | ffplay -nodisp -autoexit -

# Generate and download
ssh tts@host "tts custom-voice 'Check this out' --speaker Aiden --language English"
ssh tts@host "get output.wav" > output.wav

# List what's in the work dir
ssh tts@host "ls"

# Clean up
ssh tts@host "rm output.wav"

# Organize reference voices
ssh tts@host "mkdir refs"
ssh tts@host "put refs/neutral.wav" < my_neutral.wav
ssh tts@host "put refs/happy.wav" < my_happy.wav
```

## SSH Client Config

```
Host tts
    HostName your-server
    Port 2222
    User tts
```

Then just: `ssh tts "tts list-speakers"`

## Memory

- **0.6B float32**: ~2.4GB weights + overhead - fits in 4GB
- **1.7B float32**: ~7GB weights - needs 10GB+
- **1.7B bfloat16 (GPU)**: ~3.5GB weights - fits in 6GB VRAM
- **float16 on CPU**: don't. it produces inf/nan garbage

## Building

```
make build
make test    # build + run integration tests
```

## TODO

- [ ] NVIDIA GPU support (CUDA base image, GPU torch wheels, `--gpus` runtime)

## License

This project is licensed under [WTFPL](LICENSE) - Do What The Fuck You Want To Public License.
