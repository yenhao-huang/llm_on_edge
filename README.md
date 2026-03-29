# llm-on-iphone

Run LLMs natively on iOS and macOS devices using [ExecuTorch](https://pytorch.org/executorch/).

## Architecture

```
HuggingFace model → optimum-executorch → ExecuTorch (.pte) → etLLM iOS/macOS app
```

## Setup

```bash
uv venv llm_on_ios --python 3.12
pip install -r requirements.txt
cd optimum-executorch
pip install '.[dev]'

# or activate existing venv
source /Users/yenhaohuang/Desktop/python-venvs/llm_on_ios/bin/activate
```

## Project Structure

| Directory | Description |
|-----------|-------------|
| `export_to_executorch/` | Scripts to export Qwen3-0.6B to ExecuTorch format |
| `apple/` | etLLM Xcode project for iOS and macOS |
| `qwen3_0.6b_executorch/` | Output directory for exported model (`.pte`) |
| `test/` | Tests |

## How to Use

### 1. Export model to ExecuTorch

Download and export Qwen3-0.6B with XNNPACK backend and 8-bit quantization:

```bash
bash export_to_executorch/download_qwen.sh
```

Output: `qwen3_0.6b_executorch/model.pte`

### 2. Validate the export

```bash
python export_to_executorch/validate.py
```

### 3. Rename to app-expected filename

```bash
mv qwen3_0.6b_executorch/model.pte qwen3_0.6b_executorch/qwen3-0.6b.pte
```

### 4. Copy tokenizer config into the output directory

The app requires `tokenizer.json` and `tokenizer_config.json` alongside the `.pte` file:

```bash
cp /path/to/qwen3-0.6b/tokenizer.json qwen3_0.6b_executorch/
cp /path/to/qwen3-0.6b/tokenizer_config.json qwen3_0.6b_executorch/
```

Expected contents of `qwen3_0.6b_executorch/`:
```
qwen3-0.6b.pte
tokenizer.json
tokenizer_config.json
```

### 5. Build and run the iOS/macOS app

See [apple/README.md](apple/README.md) for full Xcode setup instructions.

**Quick start:**

```bash
open apple/etLLM.xcodeproj
```

- **iOS:** Select `etLLM` target, copy `.pte` + tokenizer files to `On My iPhone > etLLM`
- **macOS:** Select `etLLM-macOS` target, use the folder button to load model and tokenizer

### Supported Models

The app supports models whose `.pte` filename starts with the corresponding prefix:

| Prefix | Model |
|--------|-------|
| `qwen3` | Qwen3 |
| `llama` | LLaMA |
| `gemma` | Gemma3 |
| `phi4` | Phi4 |
| `smollm` | SmolLM3 |
| `voxtral` | Voxtral |

Pre-exported models are available on [HuggingFace executorch-community](https://huggingface.co/executorch-community).

## Text-to-Speech (TTS)

The app includes neural TTS powered by **Qwen3-TTS** via [TTSKit](https://github.com/argmaxinc/WhisperKit). When the model is unavailable it falls back to `AVSpeechSynthesizer`.

### How it works

- **Backend:** `TTSKit` (a product of `argmaxinc/WhisperKit`), model `qwen3TTS_0_6b`
- **Model download:** ~1 GB CoreML model, downloaded automatically from HuggingFace on first `loadModel()` call
- **Fallback:** `AVSpeechSynthesizer` is used until the neural model finishes loading

### Verify TTS pipeline

```bash
cd ~/Library/Developer/Xcode/DerivedData/etLLM-*/SourcePackages/checkouts/whisperkit
swift run whisperkit-cli tts --text "Hello, TTSKit works!" --play
```

## Platform Requirements

| | iOS | macOS |
|--|-----|-------|
| Min version | iOS 17.0 | macOS 14.0 (Sonoma) |
| Chip | Any | Apple Silicon (M1+) |
| Special entitlement | `increased-memory-limit` | None |
| Image input | Camera + Photo Library | Photo Library only |
