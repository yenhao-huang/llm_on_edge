# 語音輸入整合計畫：WhisperKit STT

## Context

現有 app 已有 TTSKit (text-to-speech)，WhisperKit package 已加入 SPM 但未使用。
目標：讓使用者在 iOS chat 介面點擊麥克風按鈕 → 錄音 → 自動轉文字填入輸入框。

---

## 架構決策

- **模式：Record-then-Transcribe** — 點一下開始錄音，再點停止後轉錄。穩定且簡單。
- **模型：`openai_whisper-base`** (~145MB) — 精準度與大小的最佳平衡點，首次使用自動從 HuggingFace 下載。
- **TTS 衝突處理：** 開始錄音前先呼叫 `speechManager.stop()`。

---

## 需要修改/新增的檔案

| 檔案 | 動作 | 說明 |
|---|---|---|
| `apple/Application/SpeechRecognitionManager.swift` | 新增 | STT 管理器，含狀態機、錄音、轉錄 |
| `apple/Application/ContentView.swift` | 修改 | 新增 mic 按鈕、@StateObject、callback |
| `apple/SupportingFiles/Info.plist` | 修改 | 加入 NSMicrophoneUsageDescription |
| Xcode project（手動） | 修改 | 在 iOS target 加入 WhisperKit framework |

---

## 實作步驟

### Step 1 — Info.plist

路徑：`apple/SupportingFiles/Info.plist`

加入：
```xml
<key>NSMicrophoneUsageDescription</key>
<string>Voice input to transcribe speech into text prompts</string>
```

### Step 2 — Xcode target（手動操作）

Xcode → etLLM iOS target → Frameworks, Libraries, and Embedded Content
→ 點 `+` → 選 `whisperkit` package 下的 `WhisperKit` product → Add

> WhisperKit package 已在 SPM 中解析完畢，只差 link 到 target。

### Step 3 — 新增 SpeechRecognitionManager.swift

路徑：`apple/Application/SpeechRecognitionManager.swift`

**狀態機：**
```
idle
  └→ loadingModel(progress:)   # 首次呼叫，下載模型
        └→ readyToRecord
              └→ recording      # 點 mic 開始
                    └→ transcribing   # 再點 mic 停止
                          └→ readyToRecord  # callback 回傳文字
```

**類別設計：**
```swift
import Foundation
import AVFoundation
import WhisperKit

@MainActor
final class SpeechRecognitionManager: ObservableObject {

    enum RecordingState: Equatable {
        case idle
        case loadingModel(progress: Double)
        case readyToRecord
        case requestingPermission
        case permissionDenied
        case recording
        case transcribing
        case error(String)
    }

    @Published var recordingState: RecordingState = .idle
    var onTranscription: ((String) -> Void)?

    private var whisperKit: WhisperKit?
    private let modelName = "openai_whisper-base"

    func loadModelIfNeeded() async { ... }
    func toggleRecording() { ... }
    func stopIfRecording() async { ... }   // TTS 開始前呼叫
    func unloadModel() async { ... }       // 記憶體警告時呼叫

    private func startRecording() async { ... }
    private func stopAndTranscribe() async { ... }
    private func requestPermission() async -> Bool { ... }
}
```

**核心 WhisperKit API：**
```swift
// 載入（首次會下載模型）
let kit = try await WhisperKit(WhisperKitConfig(
    model: "openai_whisper-base",
    verbose: false, logLevel: .error,
    prewarm: false, load: true, download: true
))

// 權限（WhisperKit 內建，不需自己寫）
let granted = await AudioProcessor.requestRecordPermission()

// 錄音（WhisperKit 內部管理 AVAudioSession）
try kit.audioProcessor.startRecordingLive(callback: nil)

// 停止並取得樣本
let samples = Array(kit.audioProcessor.audioSamples)
kit.audioProcessor.stopRecording()

// 轉錄
let results = try await kit.transcribe(audioArray: samples)
let text = results.map(\.text).joined()
    .trimmingCharacters(in: .whitespacesAndNewlines)
```

**Edge cases：**
- 空錄音（立即停止）→ 跳過 transcribe，回到 readyToRecord
- 背景執行 → 監聽 `UIApplication.willResignActiveNotification`，自動停止並轉錄
- 記憶體警告 → 呼叫 `kit.unloadModels()`，下次使用再重新載入
- Simulator（無麥克風）→ requestPermission 回傳 false → 顯示 permissionDenied，不 crash

### Step 4 — 修改 ContentView.swift

**加入 @StateObject（~line 197，緊接在 speechManager 之後）：**
```swift
@StateObject private var speechRecognitionManager = SpeechRecognitionManager()
```

**在 `.onAppear` 加入（iOS section）：**
```swift
// 背景預載（讓首次點擊無需等待）
Task { await speechRecognitionManager.loadModelIfNeeded() }

// 轉錄完成後填入 prompt
speechRecognitionManager.onTranscription = { transcribed in
    prompt = transcribed
    textFieldFocused = false
}
```

**Mic 按鈕位置：camera button 之後、TextField 之前**
```swift
#if os(iOS)
Button(action: {
    if speechManager.isSpeaking { speechManager.stop() }
    speechRecognitionManager.toggleRecording()
}) {
    MicButtonView(state: speechRecognitionManager.recordingState)
}
#endif
```

**`generate()` 加防護（防止錄音中誤觸送出）：**
```swift
guard speechRecognitionManager.recordingState != .recording else { return }
```

**send button disabled 條件補上：**
```swift
|| speechRecognitionManager.recordingState == .transcribing
```

### Step 5 — MicButtonView（放在 ContentView.swift 底部）

```swift
#if os(iOS)
private struct MicButtonView: View {
    let state: SpeechRecognitionManager.RecordingState
    @State private var isAnimating = false

    var body: some View {
        ZStack {
            if case .recording = state {
                Circle()
                    .stroke(Color.red.opacity(0.4), lineWidth: 2)
                    .frame(width: 34, height: 34)
                    .scaleEffect(isAnimating ? 1.4 : 1.0)
                    .opacity(isAnimating ? 0 : 1)
                    .animation(.easeOut(duration: 0.9).repeatForever(autoreverses: false),
                               value: isAnimating)
                    .onAppear { isAnimating = true }
                    .onDisappear { isAnimating = false }
            }
            Image(systemName: iconName)
                .resizable().scaledToFit()
                .frame(width: 24, height: 24)
                .foregroundColor(iconColor)
        }
        .frame(width: 34, height: 34)
    }

    private var iconName: String {
        switch state {
        case .recording:         return "mic.fill"
        case .transcribing:      return "waveform"
        case .loadingModel:      return "arrow.down.circle"
        case .permissionDenied:  return "mic.slash"
        case .error:             return "exclamationmark.triangle"
        default:                 return "mic"
        }
    }

    private var iconColor: Color {
        switch state {
        case .recording:    return .red
        case .transcribing: return .orange
        default:            return .primary
        }
    }
}
#endif
```

---

## 驗證方式

| 測試項目 | 預期行為 |
|---|---|
| 首次點擊 mic | 系統跳出麥克風權限對話框 |
| 模型下載 | mic 圖示短暫顯示 `arrow.down.circle` |
| 錄音中 | 圖示變紅 `mic.fill` + 脈衝動畫圓環 |
| 點停止 | 橙色 `waveform` → prompt 自動填入 |
| TTS 播放中點 mic | TTS 停止，立即開始錄音 |
| 立即點停（空錄音） | 無 crash，恢復正常狀態 |
| 切到背景 | 自動停止錄音並轉錄 |

> **注意：** 麥克風功能需在實體 iOS 裝置測試，Simulator 無法錄音。
