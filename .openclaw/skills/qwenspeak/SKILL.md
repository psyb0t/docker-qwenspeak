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

Qwen3-TTS text-to-speech over SSH. Uses a Python wrapper that whitelists commands - no shell access, no injection, no bullshit.

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
| `custom-voice` | Generate speech using one of 9 preset speakers               |
| `voice-design` | Generate speech by describing the voice in natural language   |
| `voice-clone`  | Clone a voice from a reference audio file                    |
| `list-speakers`| List available preset speakers                               |
| `tokenize`     | Encode/decode audio through the speech tokenizer             |

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

## Usage Examples

### Custom voice - preset speakers

```bash
# Basic generation
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts custom-voice 'Hello world' --speaker Ryan --language English"

# With emotion control (1.7B model only)
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts custom-voice 'I cannot believe this!' --speaker Vivian --instruct 'Speak angrily'"

# Smaller model
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts custom-voice 'Hello' --speaker Aiden --model-size 0.6b"

# Download result
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "get output.wav" > output.wav
```

### Voice design - describe the voice

```bash
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts voice-design 'Welcome to our store.' --instruct 'A warm, friendly young female voice with a cheerful tone'"

ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts voice-design 'Breaking news today.' --instruct 'Deep authoritative male news anchor voice'"
```

### Voice clone - clone any voice

```bash
# Upload reference audio
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "put ref.wav" < my_voice.wav

# Clone with transcript (ICL mode - best quality)
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts voice-clone 'New text in my voice' --ref-audio /work/ref.wav --ref-text 'What I said in the ref clip'"

# Clone without transcript (x-vector only)
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts voice-clone 'New text in my voice' --ref-audio /work/ref.wav --x-vector-only"

# Smaller model
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts voice-clone 'Hello' --ref-audio /work/ref.wav --x-vector-only --model-size 0.6b"
```

### Emotion trick for cloned voices

Record yourself with different emotions and use the matching reference file:

```bash
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "mkdir refs"
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "put refs/happy.wav" < me_happy.wav
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "put refs/angry.wav" < me_angry.wav

ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts voice-clone 'Great news!' --ref-audio /work/refs/happy.wav --ref-text 'transcript'"
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts voice-clone 'This is unacceptable' --ref-audio /work/refs/angry.wav --ref-text 'transcript'"
```

### List speakers

```bash
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts list-speakers"
```

### Generate and play immediately

```bash
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "tts custom-voice 'Hello there' --speaker Ryan"
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "get output.wav" | ffplay -nodisp -autoexit -
```

### File management

```bash
# List files
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "ls"
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "ls --json"

# Clean up
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "rm output.wav"
ssh -p $QWENSPEAK_PORT tts@$QWENSPEAK_HOST "rrmdir refs"
```

## Generation Options

| Flag                   | Default      | Description                                                  |
| ---------------------- | ------------ | ------------------------------------------------------------ |
| `--output`, `-o`       | `output.wav` | Output file path (relative to /work)                         |
| `--language`, `-l`     | `Auto`       | Language: Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| `--model-size`         | `1.7b`       | Model size: 1.7b or 0.6b                                    |
| `--device`             | `cpu`        | Device: cpu, cuda:0, etc.                                    |
| `--dtype`              | `float32`    | Model dtype: float32, float16, bfloat16 (float16/bfloat16 GPU only) |
| `--temperature`        | `0.9`        | Sampling temperature                                         |
| `--top-k`              | `50`         | Top-k sampling                                               |
| `--top-p`              | `1.0`        | Top-p / nucleus sampling                                     |
| `--repetition-penalty` | `1.05`       | Repetition penalty                                           |
| `--max-new-tokens`     | `2048`       | Max codec tokens to generate                                 |
| `--no-sample`          | off          | Greedy decoding                                              |
| `--streaming`          | off          | Streaming mode (lower latency)                               |

## Security Notes

- No shell access - all commands go through a Python wrapper
- Whitelist only - unlisted commands are rejected
- No injection - `&&`, `;`, `|`, `$()` are treated as literal arguments (no shell involved)
- SSH key auth only - no passwords
- All forwarding disabled
- All file paths locked to `/work`
