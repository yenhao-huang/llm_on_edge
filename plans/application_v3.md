# Application 架構 v3（已完成）

## 完成日期
2026-04-05

## 完成狀態
- ✅ ContentView 拆分為 ContentView + ChatViewModel（1380 行 → ~520 + ~280 行）
- ✅ Image+Extensions.swift 從 ContentView 抽出
- ✅ MicButtonView.swift 從 ContentView 抽出
- ✅ 全 project 目錄分層（移動 15 個檔案，建立 6 個子目錄）
- ✅ Xcode project.pbxproj group 結構更新
- ✅ LLMGeneration.swift 新增（Services 層，ChatViewModel 改用 llmService）
- ✅ iOS + macOS 兩個 target 均 BUILD SUCCEEDED

---

## 最終目錄結構

```
Application/
├── App.swift
├── Core/
│   └── Constants.swift
├── Features/
│   ├── Chat/
│   │   ├── ContentView.swift       ← UI shell，~520 行，@StateObject ChatViewModel
│   │   ├── ChatViewModel.swift     ← @MainActor ObservableObject，~280 行，call llmService
│   │   ├── MessageListView.swift
│   │   └── MessageView.swift
│   └── Logs/
│       ├── LogManager.swift
│       └── LogView.swift
├── Models/
│   └── Message.swift
├── Services/
│   ├── LLMGeneration.swift         ← RunnerHolder, ModelType, LLMGenerationService
│   ├── SpeechManager.swift
│   ├── SpeechRecognitionManager.swift
│   ├── ResourceManager.swift
│   └── ResourceMonitor.swift
└── UI/
    └── Components/
        ├── MicButtonView.swift
        ├── ImagePicker.swift
        └── Image+Extensions.swift
```

---

## 關鍵設計決策

### ChatViewModel 必須是 @MainActor
`SpeechManager` 和 `SpeechRecognitionManager` 都是 `@MainActor final class`，
其 `init()` 是 `@MainActor`-isolated。ChatViewModel 必須整個標註 `@MainActor`
才能在 `init()` 中直接建構這兩個 service，以及在 `generate()` 中存取它們的 property。

```swift
@MainActor
class ChatViewModel: ObservableObject { ... }
```

### Nested ObservableObject 觀察問題
SwiftUI 不自動觀察 ObservableObject 內持有的另一個 ObservableObject 的變化。
解法：在 `setupObservation()` 中用 Combine 把所有 nested service 的
`objectWillChange` forward 給 ChatViewModel：

```swift
private var cancellables = Set<AnyCancellable>()

private func setupObservation() {
    speechManager.objectWillChange
        .sink { [weak self] _ in self?.objectWillChange.send() }
        .store(in: &cancellables)
    // ... resourceManager, resourceMonitor, logManager 同上
}
```

### Live Conversation 重啟邏輯移入 ViewModel
原本在 ContentView `macOSBody` 的 `onChange` handler（iOS 完全缺少），
改用 Combine 在 ViewModel 統一處理，iOS/macOS 共用：

```swift
$isGenerating
    .dropFirst()
    .sink { [weak self] isGenerating in
        guard let self, !isGenerating,
              speechRecognitionManager.isLiveMode, !autoSpeak else { return }
        Task {
            try? await Task.sleep(nanoseconds: 500_000_000)
            self.speechRecognitionManager.startLiveModeListening()
        }
    }
    .store(in: &cancellables)

speechManager.$isSpeaking
    .dropFirst()
    .sink { [weak self] isSpeaking in
        guard let self, !isSpeaking,
              speechRecognitionManager.isLiveMode, !isGenerating else { return }
        Task {
            try? await Task.sleep(nanoseconds: 500_000_000)
            self.speechRecognitionManager.startLiveModeListening()
        }
    }
    .store(in: &cancellables)
```

---

## Xcode project.pbxproj 變更摘要

新增 6 個 PBXGroup（UUID 前綴 `GRP0000`）：
- `GRP0000CORE000000000001` — path = `Core`
- `GRP0000FEATCHAT00000001` — path = `Features/Chat`
- `GRP0000FEATLOGS00000001` — path = `Features/Logs`
- `GRP0000MODELS000000001`  — path = `Models`
- `GRP0000SERVICES0000001`  — path = `Services`
- `GRP0000UICOMPONENTS001`  — path = `UI/Components`

Application group children 縮減為 App.swift + 6 個子 group。
Build phase 條目（PBXBuildFile UUID）完全不變，只有 group 組織調整。

---

## LLMGenerationService 設計

### 職責
- 持有 `RunnerHolder`（`textRunner` / `multimodalRunner`）和 `runnerQueue`
- `loadModelIfNeededSync(modelPath:tokenizerPath:) throws -> TimeInterval?`
  - 回傳 `nil` = 已載入，不需重新載入
  - 回傳 `TimeInterval` = 本次載入耗時
  - throws = 載入失敗
- `generate(text:modelPath:thinkingMode:image:onEvent:shouldStop:) throws`
  - 透過 `onEvent: (GenerationEvent) -> Void` 回調（從 runnerQueue 呼叫，caller 負責 dispatch to main）
  - `shouldStop: () -> Bool` closure 讓 ViewModel 注入停止條件
- `resetRunners()` — 路徑改變時清空 runner

### GenerationEvent
```swift
enum GenerationEvent {
  case tokens(text: String, count: Int)
  case thinkStart(pendingText: String, pendingCount: Int)  // 觸發 "Thinking..." UI
  case thinkEnd(pendingText: String, pendingCount: Int)    // 觸發 "Finished thinking." UI
}
```

### ChatViewModel 改動
- 移除 `runnerHolder`, `runnerQueue`, `shouldStopShowingToken`
- 移除 `RunnerHolder`, `ModelType`（已搬到 LLMGeneration.swift）
- 新增 `let llmService = LLMGenerationService()`
- `generate()` 中 `capturedImage = selectedImage` 在 main thread 取值，再 dispatch 到 `llmService.runnerQueue`
- 新增 `handleGenerationEvent(_:)` 處理 GenerationEvent → 更新 messages
- `loadModelIfNeededAsync()` 移除 `reportToUI` 參數（不再需要）

---

## 未完成（v2 計劃中提及但尚未實作）

| 項目 | 說明 |
|------|------|
| ChatView+iOS.swift / ChatView+macOS.swift 拆分 | ContentView 仍為單一檔案，`#if os(iOS)` 條件編譯在同一檔案內 |
| 檔案改名（TextToSpeechService 等） | 保留原始類別名稱，只移動目錄 |
