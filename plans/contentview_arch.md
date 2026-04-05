# ContentView.swift 架構分析

## 整體結構

```
ContentView.swift
├── 圖片工具 Extension (UIImage / NSImage)
├── RunnerHolder (ObservableObject)
├── ContentView (主 View)
│   ├── State 管理
│   ├── ModelType enum
│   ├── body (macOS / iOS 平台分支)
│   ├── generate() — 核心推理邏輯
│   ├── loadModelIfNeededSync/Async()
│   └── handleFileImportResult()
```

---

## 1. 圖片工具 (L26-L180)

`UIImage` (iOS) / `NSImage` (macOS) 的 extension，提供：
- `centerCropped(to:)` — 正方形裁切
- `rgbBytes()` / `rgbBytesNormalized()` — pixel buffer 萃取（CHW 格式）
- `asImage()` / `asNormalizedImage()` — 轉換為 `ExecuTorchLLM.Image`

---

## 2. RunnerHolder (L19-L22)

```swift
class RunnerHolder: ObservableObject {
  var textRunner: TextRunner?              // 純文字模型
  var multimodalRunner: MultimodalRunner?  // 多模態模型
}
```

持有 ExecuTorch runner 的容器，生命週期跟 View 綁定。

---

## 3. State 管理 (L182-L207)

| State | 用途 |
|---|---|
| `messages` | 聊天記錄 |
| `isGenerating` | 推理進行中 |
| `shouldStopGenerating` | 停止信號 |
| `thinkingMode` | Qwen3 思考模式開關 |
| `autoSpeak` | 推理完自動 TTS |
| `speechManager` | TTS (SpeechManager) |
| `speechRecognitionManager` | ASR (SpeechRecognitionManager) |
| `runnerHolder` | 模型 runner |

---

## 4. ModelType enum (L214-L243)

支援 7 種模型：`gemma3`, `llama`, `llava`, `qwen3`, `phi4`, `smollm3`, `voxtral`
透過 filename prefix 自動判斷類型。

---

## 5. UI 分支 — macOS vs iOS

**macOS** (L272-L540)：`NavigationSplitView`，左側 sidebar 做設定，右側聊天區。

**iOS** (L544-L761)：`NavigationView` + `ZStack`，settings panel 折疊在上方。

---

## 6. generate() — 核心推理邏輯 (L789-L1158)

```
generate()
├── 格式化 prompt（依 modelType 套用對應 template）
├── runnerQueue.async（背景執行緒）
│   ├── loadModelIfNeededSync()  ← lazy load model
│   ├── multimodalRunner (llava / gemma3 / voxtral)
│   │   └── 帶圖片或音訊 input
│   └── textRunner (llama / qwen3 / phi4 / smollm3)
│       └── 處理 <think>/<\/think> token (Qwen3 thinking mode)
└── token streaming → DispatchQueue.main.async 更新 messages[]
```

Token 以每 2 個為一批 flush 到 UI，並計算 tokens/sec。

---

## 7. Live Conversation 流程

```
ASR 錄音 → onTranscription → generate() → TTS 播放
    ↑___________重新開始監聽___________________________↑
```

透過 `.onChange(of: isGenerating)` 和 `.onChange(of: speechManager.isSpeaking)` 串接 ASR ↔ LLM ↔ TTS 的循環。

---

## 8. Model 載入機制

- **Lazy load**：首次 generate 或輸入文字時觸發
- `loadModelIfNeededAsync()` — 非同步（背景預載）
- `loadModelIfNeededSync()` — 同步（generate 時確保已載入）
- 切換模型時 runner 置 nil，強制重新載入
