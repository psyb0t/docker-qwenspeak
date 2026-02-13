---
name: qwenspeak
description: Text-to-speech via a locked-down SSH container with Qwen3-TTS - preset voices, voice cloning, voice design
homepage: https://github.com/psyb0t/docker-qwenspeak
user-invocable: true
metadata:
  {
    "openclaw":
      {
        "emoji": "🗣️",
        "primaryEnv": "QWENSPEAK_HOST",
        "always": false,
      },
  }
---

# qwenspeak

## Setup Required

This skill requires `QWENSPEAK_HOST` and `QWENSPEAK_PORT` environment variables pointing to a running qwenspeak instance.

**Configure OpenClaw** (`~/.openclaw/openclaw.json`):

```json
{
  "skills": {
    "entries": {
      "qwenspeak": {
        "env": {
          "QWENSPEAK_HOST": "localhost",
          "QWENSPEAK_PORT": "2222"
        }
      }
    }
  }
}
```

Or set the environment variables directly:

```bash
export QWENSPEAK_HOST=localhost
export QWENSPEAK_PORT=2222
```

---

Qwen3-TTS text-to-speech over SSH. YAML-driven pipeline - batch multiple generations across different models in one config. Uses a Python wrapper that whitelists commands - no shell access, no injection, no bullshit.

## First Connection

Before running any commands, you must accept the host key so it gets added to `known_hosts`. Run a simple `ls` and accept the fingerprint:

```bash
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "ls"
```

If this is the first time connecting, SSH will prompt to verify the host key. Type `yes` to accept. This only needs to happen once per host. If you skip this step, all subsequent SSH commands will fail with a host key verification error.

## How It Works

All commands are executed via SSH against the qwenspeak container. The container forces every connection through a Python wrapper that only allows whitelisted commands. All file paths are locked to `/work` inside the container.

**SSH command format:**

```bash
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "<command> [args]"
```

## TTS Command

| Command | Binary               | Description        |
| ------- | -------------------- | ------------------ |
| `tts`   | `/usr/local/bin/tts` | Text-to-speech generation |

### Subcommands

| Subcommand     | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| `print-yaml`   | Print a template YAML config to stdout                       |
| `list-speakers`| List available preset speakers                               |
| `log`          | View TTS logs (`-f` to follow, `-n N` for line count)        |
| `tokenize`     | Encode/decode audio through the speech tokenizer             |

When called with no subcommand, `tts` reads a YAML config from stdin and runs the pipeline.

## YAML Pipeline

All generation is done through YAML configs piped via stdin. Get a template, fill it in, pipe it back.

```bash
# Get the YAML template
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts print-yaml" > job.yaml

# Edit it locally, then run it
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts" < job.yaml

# Download results
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "get hello.wav" > hello.wav
```

### YAML Structure

Each config has global settings and a list of steps. Each step loads a model, runs all its generations, then unloads it. Settings cascade: global → step → generation.

```yaml
device: cpu
dtype: float32
models_dir: /models
temperature: 0.9

steps:
  # custom-voice: preset speakers with optional emotion control
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

  # voice-design: describe the voice in natural language
  - mode: voice-design
    generate:
      - text: "Welcome to our store."
        instruct: "A warm, friendly young female voice with a cheerful tone"
        output: welcome.wav

  # voice-clone: clone a voice from reference audio
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

Record yourself with different emotions and use separate steps:

```bash
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "mkdir refs"
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "put refs/happy.wav" < me_happy.wav
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "put refs/angry.wav" < me_angry.wav
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

## Logging

All pipeline output is logged to `/var/log/tts/`. View logs via SSH:

```bash
# View last 20 lines
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts log"

# Follow (like tail -f)
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts log -f"

# Last 100 lines
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts log -n 100"
```

## File Operations

All paths are relative to `/work`. Traversal attempts are blocked. Absolute paths get remapped under `/work`.

| Command  | Description                                       | Example                                    |
| -------- | ------------------------------------------------- | ------------------------------------------ |
| `ls`     | List `/work` or a subdirectory (`ls -alph` style, use `--json` for JSON) | `ls` or `ls --json subdir` |
| `put`    | Upload file from stdin                            | `put ref.wav` (pipe file via stdin)        |
| `get`    | Download file to stdout                           | `get output.wav` (redirect stdout to file) |
| `rm`     | Delete a file (not directories)                   | `rm old.wav`                               |
| `mkdir`  | Create directory (recursive)                      | `mkdir refs`                               |
| `rmdir`  | Remove empty directory                            | `rmdir refs`                               |
| `rrmdir` | Remove directory and everything in it recursively | `rrmdir refs`                              |

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

All of these can be set at global, step, or generation level. Lower levels override higher ones.

| Field                | Default   | Description                                                  |
| -------------------- | --------- | ------------------------------------------------------------ |
| `device`             | `cpu`     | Device: cpu, cuda:0, etc.                                    |
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
| `text`               | required  | Generation: text to synthesize                               |
| `output`             | required  | Generation: output file path (relative to /work)             |
| `speaker`            | `Vivian`  | custom-voice: speaker name                                   |
| `language`           | `Auto`    | Language for synthesis                                       |
| `instruct`           | -         | custom-voice: emotion/style; voice-design: voice description |
| `ref_audio`          | -         | voice-clone: reference audio file path                       |
| `ref_text`           | -         | voice-clone: transcript of reference audio                   |
| `x_vector_only`      | `false`   | voice-clone: use speaker embedding only                      |

### List speakers

```bash
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts list-speakers"
```

### File management

```bash
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "ls"
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "ls --json"
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "rm old.wav"
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "rrmdir refs"
```

## Security Notes

- No shell access - all commands go through a Python wrapper
- Whitelist only - unlisted commands are rejected
- No injection - `&&`, `;`, `|`, `$()` are treated as literal arguments (no shell involved)
- SSH key auth only - no passwords
- All forwarding disabled
- All file paths locked to `/work`
