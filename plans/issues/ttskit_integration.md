# TTSKit 整合 Issues（2026-03-28）

## Issue 1：自己刻 TTSKit 而不是用現成 library

**症狀**
Build failed — `Unable to find module dependency: 'TTSKit'`

**原因**
`SpeechManager.swift` 已寫好 `import TTSKit`，沒先查就直接新建了本地 Swift Package（`apple/TTSKit/`）來讓 build 通過。

**正確解法**
`TTSKit` 是 `argmaxinc/WhisperKit` repo 裡的 library product（同 repo 還有 `WhisperKit`、`SpeakerKit`）。`whisperkit` 已在 `Package.resolved`，只需在 `project.pbxproj` 加：
- `XCSwiftPackageProductDependency`（productName = TTSKit，package 指向已有的 `XCRemoteSwiftPackageReference "whisperkit"`）
- 兩個 target（iOS + macOS）的 Frameworks build phase 各加一筆 build file

**教訓**：遇到 `import XxxKit` build 失敗，先進 GitHub README 確認有沒有現成 library product，再決定要不要自建。

---

## Issue 2：SpeechManager init API 與真實 TTSKit 不符

**症狀**
自建 stub 用 `TTSKit(model: .qwen3TTS_0_6b, verbose: false)`，但真實 TTSKit API 是：
```swift
TTSKit(TTSKitConfig(model: .qwen3TTS_0_6b))
```

**解法**
換成 `TTSKitConfig` wrapper，移除 `modelState` property（改用 optional `tts != nil` 判斷載入狀態）。

---

## Issue 3：ContentView 仍呼叫已移除的 SpeechManager method

**症狀**
```
error: value of type 'SpeechManager' has no member 'requestPersonalVoiceAccessIfAvailable'
```

**原因**
舊版 `SpeechManager` 用 `AVSpeechSynthesizer` + Personal Voice，有授權請求方法。新版改用 TTSKit 後這個 method 不存在。

**解法**
`ContentView.swift:659`：
```swift
// Before
Task { await speechManager.requestPersonalVoiceAccessIfAvailable() }
// After
Task { await speechManager.loadModel() }
```

---

## Issue 4：TTSKit 首次使用需下載 ~1GB CoreML 模型

**說明**
`TTSKit(TTSKitConfig(model: .qwen3TTS_0_6b))` 首次執行時從 HuggingFace 下載 Qwen3-TTS CoreML 模型（約 1GB）。Simulator 首次 launch 後 `loadModel()` 會自動觸發，需等幾分鐘才能 speak。

**驗證方式**
用 whisperkit DerivedData checkout 的 CLI 在 macOS 上驗證整條 pipeline：
```bash
cd ~/Library/Developer/Xcode/DerivedData/etLLM-*/SourcePackages/checkouts/whisperkit
swift run whisperkit-cli tts --text "Hello, TTSKit works!" --play
```
輸出 `output.m4a`（53KB，2.48 秒）→ CoreML 推理 + AVAudioEngine 播放全程無 error。

---

## Issue 5：TTSKit 在 iOS Simulator 上 crash

**症狀**
```
Fatal error: There is no uniform distribution on an infinite range
```
crash 在 `TTSKit/Utilities/Sampling/GreedyTokenSampler.swift:239`：
```swift
let randomValue = Float.random(in: 0..<probSum, using: &rng)
```

**根本原因**
iOS Simulator 沒有 ANE（Apple Neural Engine），CoreML 只能走 CPU path。
Qwen3-TTS CoreML 模型在 CPU 路徑下輸出 NaN 或全零 logits，
`probsArray.reduce(0, +)` 算出 `probSum = 0`（或 NaN），
`Float.random(in: 0..<0)` 觸發 fatal error。
TTSKit 未對 `probSum <= 0` 做 guard，屬 TTSKit 的 bug（已回報路徑：argmaxinc/WhisperKit）。

**macOS CLI 可以的原因**
macOS M 系列晶片有 ANE，CoreML 走加速路徑，logits 正常。

**結論：TTSKit 必須在真機上測試，Simulator 不支援。**

whisperkit 版本：commit `3817d28`（2026-03-28 最新），尚未修復。
