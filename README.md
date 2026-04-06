# llm-on-iphone

Run LLMs natively on iOS and macOS with voice chat support.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 View Layer                                  │
│  ContentView · MessageView · MIC/Image                      │
├─────────────────────────────────────────────────────────────┤
│                 Chat ViewModel                              │
├────────────────────────────────--───────────────────────────┤
│                 Service Layer                               │
│  ASR         │  LLM Generation  │  TTS                      │
│  WhisperKit  │  LLMGeneration   │  TTSKit / AVSpeech        │
├──────────────┴──────────────────┴───────────────────────────┤
│                 Model Layer                                 │
│          ExecuTorch · optimum-executorch · CoreML           │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Text chat** — multi-turn conversation with on-device LLM inference
- **Voice chat** — full voice pipeline: speak → transcribe (ASR) → generate reply (LLM) → speak back (TTS)

### Services

| Service | Implementation | Notes |
|---------|---------------|-------|
| ASR | [WhisperKit](https://github.com/argmaxinc/WhisperKit) — `openai_whisper-large-v3` | Silence detection, live transcription |
| LLM | ExecuTorch `.pte` model via C++ runner | XNNPACK backend, 8-bit quantization |
| TTS | [TTSKit](https://github.com/argmaxinc/WhisperKit) — `qwen3TTS_0_6b` | Falls back to `AVSpeechSynthesizer` while loading |

### Supported LLM Models

| Model file prefix | Model |
|-------------------|-------|
| `qwen3` | Qwen3-0.6B / Qwen3-8B |
| `qwen35` | Qwen3.5 |
| `llama` | LLaMA |
| `gemma` | Gemma3 |
| `phi4` | Phi4 |
| `smollm` | SmolLM3 |

## Project Structure

```
llm-on-iphone/
├── apple/
│   └── Application/
│       ├── Features/Chat/       # UI + ChatViewModel
│       ├── Services/
│       │   ├── LLMGeneration.swift          # ExecuTorch inference
│       │   ├── SpeechManager.swift          # TTS (TTSKit)
│       │   └── SpeechRecognitionManager.swift # ASR (WhisperKit)
│       └── Models/
├── export_to_executorch/        # Python scripts to export models to .pte
│   ├── export_qwen_0.6b.sh
│   ├── export_qwen_8b.sh
│   └── export_qwen35.sh
└── plans/                       # Architecture and design docs
```

## Requirements

### Platform

| | iOS | macOS |
|--|-----|-------|
| Min version | iOS 17.0 | macOS 14.0 (Sonoma) |
| Chip | Any | Apple Silicon (M1+) |
| Special entitlement | `increased-memory-limit` | None |

### Environment (for model export)

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (recommended)
- ExecuTorch + optimum-executorch

```bash
uv venv llm_on_ios --python 3.12
source llm_on_ios/bin/activate
pip install -r requirements.txt
pip install './optimum-executorch[dev]'
```

## How to Use

### 1. Export a model to ExecuTorch

**Qwen3-0.6B** (recommended for quick start):
```bash
bash export_to_executorch/export_qwen_0.6b.sh
```

**Qwen3-8B:**
```bash
bash export_to_executorch/export_qwen_8b.sh
```

**Qwen3.5** (requires cloning the ExecuTorch repo first):
```bash
git clone org-21003710@github.com:pytorch/executorch.git
bash export_to_executorch/export_qwen35.sh
```

Output: `qwen3_0.6b_executorch/model.pte`

### 2. Validate the export

```bash
python export_to_executorch/validate.py
```

### 3. Prepare model files

Rename the exported model and copy tokenizer files:

```bash
mv qwen3_0.6b_executorch/model.pte qwen3_0.6b_executorch/qwen3-0.6b.pte
cp /path/to/qwen3-0.6b/tokenizer.json qwen3_0.6b_executorch/
cp /path/to/qwen3-0.6b/tokenizer_config.json qwen3_0.6b_executorch/
```

Expected contents:
```
qwen3_0.6b_executorch/
├── qwen3-0.6b.pte
├── tokenizer.json
└── tokenizer_config.json
```

> Pre-exported models are also available on [HuggingFace executorch-community](https://huggingface.co/executorch-community).

### 4. Build and run the app

```bash
open apple/etLLM.xcodeproj
```

- **iOS:** Select the `etLLM` target. Copy `.pte` + tokenizer files to `On My iPhone > etLLM`.
- **macOS:** Select the `etLLM-macOS` target. Use the folder button in the app to load the model directory.

ASR and TTS models (~1 GB each) are downloaded automatically from HuggingFace on first launch.
