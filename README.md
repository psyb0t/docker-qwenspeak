# docker-qwenspeak

[![Docker Hub](https://img.shields.io/docker/v/psyb0t/qwenspeak?sort=semver&label=Docker%20Hub)](https://hub.docker.com/r/psyb0t/qwenspeak)

Qwen3-TTS text-to-speech over SSH. Pick a voice, clone a voice, design a voice - all through a YAML config piped via stdin. Models run locally, no API keys, no cloud bullshit.

Built on top of [psyb0t/lockbox](https://github.com/psyb0t/docker-lockbox) - see that repo for the security model, file operations, path sandboxing, and all the SSH lockdown details.

## Features

- **YAML pipeline** - batch multiple generations across different models in one config
- **9 premium speakers** - male/female voices across Chinese, English, Japanese, Korean
- **Emotion/style control** - make any preset speaker happy, angry, sad, whatever
- **Voice design** - describe the voice you want in plain English and it generates it
- **Voice cloning** - clone any voice from a 3-second audio sample
- **10 languages** - Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **CPU & GPU** - runs on CPU by default, NVIDIA GPU via `--processing-unit cuda`

## Models

You need to download models locally before running. Pick what you need:

```bash
pip install -U "huggingface_hub[cli]"

# Speech tokenizer (used by all models)
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

### install.sh (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/psyb0t/docker-qwenspeak/main/install.sh | sudo bash
```

This sets up `~/.qwenspeak/` with the docker-compose file, authorized_keys, and work directory, then drops a `qwenspeak` command into `/usr/local/bin`.

Add your SSH key, point it at your models, and start it:

```bash
cat ~/.ssh/id_rsa.pub >> ~/.qwenspeak/authorized_keys
qwenspeak start -d -m /path/to/your/models
```

```bash
qwenspeak start -d                        # foreground or detached
qwenspeak start -d --port 2223             # custom port (default 2222)
qwenspeak start -d -m /mnt/hdd/models     # custom models directory
qwenspeak start -d --processing-unit cuda                # GPU mode (requires NVIDIA Container Toolkit)
qwenspeak start -d --processing-unit cuda --gpus 0       # use only GPU 0
qwenspeak start -d --processing-unit cuda --gpus 0,1     # use GPUs 0 and 1
qwenspeak start -d --memory 4g --swap 2g --cpus 4  # 4GB RAM, 2GB swap, 4 CPUs
qwenspeak stop                             # stop
qwenspeak upgrade                          # pull latest image, asks to stop/restart if running
qwenspeak uninstall                        # stop and remove everything
qwenspeak status                           # show status
qwenspeak logs                             # show logs
```

All flags persist to `~/.qwenspeak/.env` - next `start` reuses the last values.

### docker run

```bash
docker pull psyb0t/qwenspeak

cat ~/.ssh/id_rsa.pub > authorized_keys
mkdir -p work host_keys logs

docker run -d \
  --name qwenspeak \
  --restart unless-stopped \
  --memory 4g \
  -p 2222:22 \
  -e "LOCKBOX_UID=$(id -u)" \
  -e "LOCKBOX_GID=$(id -g)" \
  -e "TTS_LOG_RETENTION=7d" \
  -v $(pwd)/authorized_keys:/etc/lockbox/authorized_keys:ro \
  -v $(pwd)/host_keys:/etc/lockbox/host_keys \
  -v $(pwd)/work:/work \
  -v $(pwd)/logs:/var/log/tts \
  -v /path/to/your/models:/models:ro \
  psyb0t/qwenspeak

ssh -p 2222 tts@localhost "tts list-speakers"
```

### GPU (NVIDIA)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

```bash
docker run -d \
  --name qwenspeak \
  --restart unless-stopped \
  --gpus all \
  -p 2222:22 \
  -e "LOCKBOX_UID=$(id -u)" \
  -e "LOCKBOX_GID=$(id -g)" \
  -e "PROCESSING_UNIT=cuda" \
  -v $(pwd)/authorized_keys:/etc/lockbox/authorized_keys:ro \
  -v $(pwd)/host_keys:/etc/lockbox/host_keys \
  -v $(pwd)/work:/work \
  -v $(pwd)/logs:/var/log/tts \
  -v /path/to/your/models:/models:ro \
  psyb0t/qwenspeak
```

FlashAttention-2 is included and auto-enables on GPU. It requires fp16/bf16 — if your dtype is float32, it auto-switches to bfloat16.

Device is controlled by the `PROCESSING_UNIT` env var (not in YAML). Set via `--processing-unit cuda` on the installer or `-e PROCESSING_UNIT=cuda` on docker run.

## Allowed Commands

| Command | Description                                             |
| ------- | ------------------------------------------------------- |
| `tts`   | Text-to-speech generation (the only command, that's it) |

## How It Works

All generation is driven by YAML configs piped via stdin. Jobs run asynchronously — submit a config, get a job UUID back immediately, poll for progress, download results when done. Jobs execute sequentially (one pipeline at a time), new submissions queue up automatically. Max queue size is 50 by default (`TTS_MAX_QUEUE` env var).

```bash
# Get the YAML template
ssh tts@host "tts print-yaml" > job.yaml

# Edit it
vim job.yaml

# Submit (returns immediately with job ID)
ssh tts@host "tts" < job.yaml
# {"id": "550e8400-...", "status": "queued", "total_steps": 3, "total_generations": 7}

# Check progress
ssh tts@host "tts get-job 550e8400"

# View job log
ssh tts@host "tts get-job-log 550e8400"

# Follow job log (like tail -f)
ssh tts@host "tts get-job-log 550e8400 -f"

# List all jobs
ssh tts@host "tts list-jobs"

# Cancel
ssh tts@host "tts cancel-job 550e8400"

# Download results when done
ssh tts@host "get hello.wav" > hello.wav
```

### YAML Config

Each config has global settings and a list of steps. Each step loads a model, runs all its generations, then unloads it. Settings cascade: global → step → generation.

```yaml
# Global settings
dtype: float32
models_dir: /models
flash_attn: auto           # auto-detects; set true/false to override

# Generation defaults
temperature: 0.9
top_k: 50
top_p: 1.0
repetition_penalty: 1.05
max_new_tokens: 2048
streaming: false
no_sample: false

steps:
  - mode: custom-voice
    model_size: 1.7b
    speaker: Ryan
    language: English
    generate:
      - text: "Hello world"
        output: hello.wav
      - text: "I cannot believe this!"
        speaker: Vivian
        instruct: "Speak angrily"
        output: angry.wav

  - mode: voice-design
    generate:
      - text: "Welcome to our store."
        instruct: "A warm, friendly young female voice with a cheerful tone"
        output: welcome.wav

  - mode: voice-clone
    model_size: 1.7b
    ref_audio: ref.wav
    ref_text: "Transcript of reference"
    generate:
      - text: "First line in cloned voice"
        output: clone1.wav
      - text: "Second line"
        output: clone2.wav
```

### TTS Modes

**custom-voice** - Pick from 9 preset speakers. The 1.7B model supports emotion/style control via `instruct`.

**voice-design** - Describe the voice in natural language via `instruct`. Only available as 1.7B.

**voice-clone** - Clone a voice from reference audio. Set `ref_audio` and `ref_text` at the step level to reuse the voice prompt across generations. Use `x_vector_only: true` to skip the transcript.

### Batching

The YAML pipeline loads each model once and runs all its generations before moving on. Put all custom-voice generations in one step, all voice-clone generations in another, etc.

**Emotion trick for cloned voices:** upload reference files with different emotions and use separate steps:

```bash
ssh tts@host "create-dir refs"
ssh tts@host "put refs/happy.wav" < me_happy.wav
ssh tts@host "put refs/angry.wav" < me_angry.wav
```

```yaml
steps:
  - mode: voice-clone
    ref_audio: refs/happy.wav
    ref_text: "transcript of happy ref"
    generate:
      - text: "Great news everyone!"
        output: happy1.wav
      - text: "I'm so glad to hear that"
        output: happy2.wav

  - mode: voice-clone
    ref_audio: refs/angry.wav
    ref_text: "transcript of angry ref"
    generate:
      - text: "This is unacceptable"
        output: angry1.wav
```

### Job Management

```bash
# List all jobs
ssh tts@host "tts list-jobs"
ssh tts@host "tts list-jobs --json"

# Get job details
ssh tts@host "tts get-job <uuid-or-prefix>"

# View job log
ssh tts@host "tts get-job-log <uuid-or-prefix>"

# Follow job log (like tail -f)
ssh tts@host "tts get-job-log <uuid-or-prefix> -f"

# Cancel a running or queued job
ssh tts@host "tts cancel-job <uuid-or-prefix>"
```

Job statuses: `queued` → `running` → `completed` | `failed` | `cancelled`

Jobs are retained for inspection after completion. Completed jobs are auto-cleaned after 1 day, all jobs after 1 week. You can use UUID prefixes (e.g. first 8 chars) for convenience.

### Other Subcommands

```bash
# List available speakers
ssh tts@host "tts list-speakers"

# Tokenize round-trip (encode audio → speech tokens → decode back)
ssh tts@host "tts tokenize input.wav"
```

## Logging

All pipeline and tokenize output is logged to `/var/log/tts/`. Mount this as a volume to access logs from the host.

Two files are maintained:
- `tts.log` - current log (truncated when a new day starts)
- `YYYY_MM_DD_tts.log` - daily archive

```bash
# View last 20 lines
ssh tts@host "tts log"

# View last 100 lines
ssh tts@host "tts log -n 100"

# Follow (like tail -f)
ssh tts@host "tts log -f"

# Follow with initial 50 lines
ssh tts@host "tts log -f -n 50"
```

Old daily logs are cleaned up automatically based on the `TTS_LOG_RETENTION` env var (default: `7d`). Supports `s`, `m`, `h`, `d`, `w` suffixes. Cleanup runs at the start of each pipeline execution.

## Available Speakers

| Speaker  | Gender | Language | Description                                    |
| -------- | ------ | -------- | ---------------------------------------------- |
| Vivian   | Female | Chinese  | Bright, slightly edgy young voice              |
| Serena   | Female | Chinese  | Warm, gentle young voice                       |
| Uncle_Fu | Male   | Chinese  | Seasoned, low mellow timbre                    |
| Dylan    | Male   | Chinese  | Youthful Beijing dialect, clear natural timbre |
| Eric     | Male   | Chinese  | Lively Chengdu/Sichuan dialect, slightly husky |
| Ryan     | Male   | English  | Dynamic with strong rhythmic drive             |
| Aiden    | Male   | English  | Sunny American, clear midrange                 |
| Ono_Anna | Female | Japanese | Playful, light nimble timbre                   |
| Sohee    | Female | Korean   | Warm with rich emotion                         |

## YAML Options

### Global / Step / Generation

All of these can be set at any level. Lower levels override higher ones.

| Field                | Default   | Description                                                         |
| -------------------- | --------- | ------------------------------------------------------------------- |
| `dtype`              | `float32` | Model dtype: float32, float16, bfloat16 (float16/bfloat16 GPU only) |
| `flash_attn`         | `auto`    | FlashAttention-2: auto-detects, auto-switches float32→bfloat16      |
| `temperature`        | `0.9`     | Sampling temperature                                                |
| `top_k`              | `50`      | Top-k sampling                                                      |
| `top_p`              | `1.0`     | Top-p / nucleus sampling                                            |
| `repetition_penalty` | `1.05`    | Repetition penalty                                                  |
| `max_new_tokens`     | `2048`    | Max codec tokens to generate                                        |
| `no_sample`          | `false`   | Greedy decoding                                                     |
| `streaming`          | `false`   | Streaming mode (lower latency)                                      |

### Step-only

| Field        | Default  | Description                                      |
| ------------ | -------- | ------------------------------------------------ |
| `mode`       | required | `custom-voice`, `voice-design`, or `voice-clone` |
| `model_size` | `1.7b`   | Model size: `1.7b` or `0.6b`                     |

### Generation-specific

| Field           | Used by                    | Description                                                   |
| --------------- | -------------------------- | ------------------------------------------------------------- |
| `text`          | all                        | Text to synthesize (required)                                 |
| `output`        | all                        | Output file path (required)                                   |
| `speaker`       | custom-voice               | Speaker name (default: Vivian)                                |
| `language`      | all                        | Language (default: Auto)                                      |
| `instruct`      | custom-voice, voice-design | Emotion/style instruction or voice description                |
| `ref_audio`     | voice-clone                | Reference audio file path (required)                          |
| `ref_text`      | voice-clone                | Transcript of reference audio (required unless x_vector_only) |
| `x_vector_only` | voice-clone                | Use speaker embedding only, no transcript needed              |

## File Operations

All paths are relative to the work directory. Traversal attempts are blocked.

| Command                      | Description                                             |
| ---------------------------- | ------------------------------------------------------- |
| `put <path>`                 | Upload file from stdin                                  |
| `get <path>`                 | Download file to stdout                                 |
| `list-files [path] [--json]` | List directory (`--json` for JSON output)               |
| `remove-file <path>`         | Delete a file                                           |
| `create-dir <path>`          | Create directory (recursive)                            |
| `remove-dir <path>`          | Remove empty directory                                  |
| `remove-dir-recursive <path>`| Remove directory and everything in it recursively       |
| `move-file <src> <dst>`      | Move or rename a file                                   |
| `copy-file <src> <dst>`      | Copy a file                                             |
| `file-info <path>`           | File metadata as JSON (size, modified, mode, owner)     |
| `file-exists <path>`         | Check if file exists (prints true/false)                |
| `file-hash <path>`           | SHA-256 hash of a file                                  |
| `disk-usage [path]`          | Total bytes used by file or directory tree              |
| `search-files <pattern>`     | Glob search (supports `**` recursive)                   |
| `append-file <path>`         | Append stdin to an existing file                        |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROCESSING_UNIT` | `cpu` | Device: `cpu` or `cuda` |
| `TTS_LOG_RETENTION` | `7d` | Log retention duration (`s`, `m`, `h`, `d`, `w` suffixes) |
| `TTS_MAX_QUEUE` | `50` | Max queued + running jobs before rejecting new submissions |

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

## License

This project is licensed under [WTFPL](LICENSE) - Do What The Fuck You Want To Public License.
