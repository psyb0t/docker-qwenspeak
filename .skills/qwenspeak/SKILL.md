---
name: qwenspeak
description: Text-to-speech generation via Qwen3-TTS over SSH. Preset voices, voice cloning, voice design. Use when the user wants to generate speech audio, clone voices, or work with TTS.
compatibility: Requires ssh and a running qwenspeak instance. QWENSPEAK_HOST and QWENSPEAK_PORT env vars must be set.
metadata:
  version: 1.2.0
  author: psyb0t
  homepage: https://github.com/psyb0t/docker-qwenspeak
---

# qwenspeak

YAML-driven text-to-speech over SSH using Qwen3-TTS models. All commands go through a locked-down container with command whitelisting and path sandboxing.

## Setup

Set these environment variables:

```bash
export QWENSPEAK_HOST=localhost
export QWENSPEAK_PORT=2222
```

## SSH Wrapper

Use `scripts/qwenspeak.sh` instead of raw SSH commands. It handles host, port, and host key acceptance automatically.

```bash
scripts/qwenspeak.sh <command> [args]
scripts/qwenspeak.sh <command> < input_file
scripts/qwenspeak.sh <command> > output_file
```

## TTS Generation

Jobs run asynchronously. Submit YAML, get a job UUID back immediately, poll for progress.

```bash
# Get the YAML template
scripts/qwenspeak.sh "tts print-yaml" > job.yaml

# Submit job (returns JSON with job ID immediately)
scripts/qwenspeak.sh "tts" < job.yaml
# {"id": "550e8400-...", "status": "pending", "total_steps": 3, "total_generations": 7}

# Check progress
scripts/qwenspeak.sh "tts get-job 550e8400"

# View job log
scripts/qwenspeak.sh "tts get-job-log 550e8400"

# Follow job log (like tail -f)
scripts/qwenspeak.sh "tts get-job-log 550e8400 -f"

# List all jobs
scripts/qwenspeak.sh "tts list-jobs"

# Cancel a running job
scripts/qwenspeak.sh "tts cancel-job 550e8400"

# Download result when done
scripts/qwenspeak.sh "get hello.wav" > hello.wav
```

### YAML Structure

Each config has global settings and a list of steps. Each step loads a model, runs all its generations, then unloads it. Settings cascade: global > step > generation.

```yaml
dtype: float32
models_dir: /models
temperature: 0.9

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
    ref_audio: /work/ref.wav
    ref_text: "Transcript of reference"
    generate:
      - text: "First line in cloned voice"
        output: clone1.wav
      - text: "Second line"
        output: clone2.wav
```

### Modes

**custom-voice** - Pick from 9 preset speakers. The 1.7B model supports emotion/style via `instruct`.

**voice-design** - Describe the voice in natural language via `instruct`. Only available as 1.7B.

**voice-clone** - Clone a voice from reference audio. Set `ref_audio` and `ref_text` at the step level to reuse the voice prompt across generations. Use `x_vector_only: true` to skip the transcript.

### Emotion trick for cloned voices

Upload reference files with different emotions and use separate steps:

```bash
scripts/qwenspeak.sh "create-dir refs"
scripts/qwenspeak.sh "put refs/happy.wav" < me_happy.wav
scripts/qwenspeak.sh "put refs/angry.wav" < me_angry.wav
```

```yaml
steps:
  - mode: voice-clone
    ref_audio: /work/refs/happy.wav
    ref_text: "transcript of happy ref"
    generate:
      - text: "Great news everyone!"
        output: happy1.wav

  - mode: voice-clone
    ref_audio: /work/refs/angry.wav
    ref_text: "transcript of angry ref"
    generate:
      - text: "This is unacceptable"
        output: angry1.wav
```

## Job Management

```bash
# List all jobs (shows id, status, progress, created time)
scripts/qwenspeak.sh "tts list-jobs"
scripts/qwenspeak.sh "tts list-jobs --json"

# Get full job details as JSON
scripts/qwenspeak.sh "tts get-job <uuid-or-prefix>"

# View job log
scripts/qwenspeak.sh "tts get-job-log <uuid-or-prefix>"

# Follow job log (like tail -f)
scripts/qwenspeak.sh "tts get-job-log <uuid-or-prefix> -f"

# Cancel a running job (kills the worker process immediately)
scripts/qwenspeak.sh "tts cancel-job <uuid-or-prefix>"
```

Job statuses: `pending` → `running` → `completed` | `failed` | `cancelled`

Jobs are ephemeral — cleaned up when completed/failed/cancelled and on container restart.

## Other Commands

```bash
# List available speakers
scripts/qwenspeak.sh "tts list-speakers"

# View logs (includes output from background jobs)
scripts/qwenspeak.sh "tts log"
scripts/qwenspeak.sh "tts log -f"
scripts/qwenspeak.sh "tts log -n 100"

# Tokenize round-trip
scripts/qwenspeak.sh "tts tokenize input.wav"
```

## File Operations

All paths relative to `/work`. Traversal blocked.

| Command                | Description                          | Example                                                         |
| ---------------------- | ------------------------------------ | --------------------------------------------------------------- |
| `list-files`           | List directory (`--json` for JSON)   | `scripts/qwenspeak.sh "list-files"`                                |
| `put`                  | Upload file from stdin               | `scripts/qwenspeak.sh "put ref.wav" < ref.wav`                     |
| `get`                  | Download file to stdout              | `scripts/qwenspeak.sh "get out.wav" > out.wav`                     |
| `remove-file`          | Delete a file                        | `scripts/qwenspeak.sh "remove-file old.wav"`                       |
| `create-dir`           | Create directory                     | `scripts/qwenspeak.sh "create-dir refs"`                           |
| `remove-dir`           | Remove empty directory               | `scripts/qwenspeak.sh "remove-dir refs"`                           |
| `remove-dir-recursive` | Remove directory recursively         | `scripts/qwenspeak.sh "remove-dir-recursive refs"`                 |
| `move-file`            | Move or rename a file                | `scripts/qwenspeak.sh "move-file old.wav new.wav"`                 |
| `copy-file`            | Copy a file                          | `scripts/qwenspeak.sh "copy-file src.wav dst.wav"`                 |
| `file-info`            | File metadata as JSON                | `scripts/qwenspeak.sh "file-info out.wav"`                         |
| `file-exists`          | Check if file exists (true/false)    | `scripts/qwenspeak.sh "file-exists out.wav"`                       |
| `file-hash`            | SHA-256 hash of a file               | `scripts/qwenspeak.sh "file-hash out.wav"`                         |
| `disk-usage`           | Total bytes used by file/dir         | `scripts/qwenspeak.sh "disk-usage refs"`                           |
| `search-files`         | Glob search (`**` recursive)         | `scripts/qwenspeak.sh "search-files **/*.wav"`                     |
| `append-file`          | Append stdin to existing file        | `scripts/qwenspeak.sh "append-file log.txt" < extra.txt`           |

## Available Speakers

| Speaker   | Gender | Language | Description                                     |
| --------- | ------ | -------- | ----------------------------------------------- |
| Vivian    | Female | Chinese  | Bright, slightly edgy young voice               |
| Serena    | Female | Chinese  | Warm, gentle young voice                        |
| Uncle_Fu  | Male   | Chinese  | Seasoned, low mellow timbre                     |
| Dylan     | Male   | Chinese  | Youthful Beijing dialect, clear natural timbre   |
| Eric      | Male   | Chinese  | Lively Chengdu/Sichuan dialect, slightly husky  |
| Ryan      | Male   | English  | Dynamic with strong rhythmic drive              |
| Aiden     | Male   | English  | Sunny American, clear midrange                  |
| Ono_Anna  | Female | Japanese | Playful, light nimble timbre                    |
| Sohee     | Female | Korean   | Warm with rich emotion                          |

## YAML Options

All settings can be set at global, step, or generation level. Lower levels override higher ones. Device is controlled by the `PROCESSING_UNIT` env var on the container, not in YAML.

| Field                | Default   | Description                                                  |
| -------------------- | --------- | ------------------------------------------------------------ |
| `dtype`              | `float32` | Model dtype: float32, float16, bfloat16 (float16/bfloat16 GPU only) |
| `flash_attn`         | `false`   | Use FlashAttention-2 (GPU only)                              |
| `temperature`        | `0.9`     | Sampling temperature                                         |
| `top_k`              | `50`      | Top-k sampling                                               |
| `top_p`              | `1.0`     | Top-p / nucleus sampling                                     |
| `repetition_penalty` | `1.05`    | Repetition penalty                                           |
| `max_new_tokens`     | `2048`    | Max codec tokens to generate                                 |
| `no_sample`          | `false`   | Greedy decoding                                              |
| `streaming`          | `false`   | Streaming mode (lower latency)                               |
| `mode`               | required  | Step only: `custom-voice`, `voice-design`, or `voice-clone`  |
| `model_size`         | `1.7b`    | Step only: `1.7b` or `0.6b`                                 |
| `text`               | required  | Text to synthesize                                           |
| `output`             | required  | Output file path (relative to /work)                         |
| `speaker`            | `Vivian`  | custom-voice: speaker name                                   |
| `language`           | `Auto`    | Language for synthesis                                       |
| `instruct`           | -         | custom-voice: emotion/style; voice-design: voice description |
| `ref_audio`          | -         | voice-clone: reference audio file path                       |
| `ref_text`           | -         | voice-clone: transcript of reference audio                   |
| `x_vector_only`      | `false`   | voice-clone: use speaker embedding only                      |
