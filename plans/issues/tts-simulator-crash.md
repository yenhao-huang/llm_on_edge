# Issue: TTSKit tts.play() 在 iOS Simulator 崩潰

**狀態：已知上游 bug，無法在 Simulator 使用 TTS**
**發現日期：2026-03-29**

---

## 症狀

App 在 iOS Simulator 上啟動後，TTSKit 模型載入成功，但 `tts.play()` 呼叫後立即崩潰：

```
[SpeechManager] Loading TTSKit model...
[SpeechManager] TTSKit loaded successfully
[SpeechManager] speak() called, tts=loaded, text=T T S test on iOS simulator
[SpeechManager] Starting tts.play()...
Fatal error: There is no uniform distribution on an infinite range
Swift/FloatingPointRandom.swift:62
```

---

## 環境

- Simulator: iPhone 16 (iOS 18.0, arm64)
- TTSKit: whisperkit @ main (3817d28)
- 模型: Qwen3-TTS 0.6b

---

## 崩潰位置

`TTSKit/Utilities/Sampling.swift:239`

```swift
let probSum = probsArray.reduce(0, +)
let randomValue = Float.random(in: 0..<probSum, using: &rng)  // ← 崩潰
```

---

## 根本原因

**問題鏈：**

```
Simulator → CoreML 使用 CPU backend（非 ANE/GPU）
    ↓
CPU 精度不穩定，logits 輸出含 NaN
    ↓
softmax(NaN logits) → NaN 機率值
    ↓
probsArray.reduce(0, +) → NaN（probSum = NaN）
    ↓
Float.random(in: 0..<NaN) → range 無效
    ↓
Fatal error: There is no uniform distribution on an infinite range
```

TTSKit 在 `sampleFromProbs()` 中沒有對 `probSum` 做 NaN/Inf guard。

**這是 TTSKit 上游 bug**，不是本專案的 code 問題。

---

## 實機 vs Simulator 差異

| 環境 | CoreML backend | 結果 |
|---|---|---|
| 實機 (iPhone) | Neural Engine (ANE) / GPU | 數值穩定，正常運作 |
| iOS Simulator | CPU only | NaN 傳播 → 崩潰 |

---

## 影響範圍

- TTS 功能無法在 Simulator 測試
- LLM 生成後自動 speak() 會使 Simulator 上的 App 崩潰

## 臨時解法

在 Simulator 環境下跳過 TTS：

```swift
func speak(_ text: String) {
    #if targetEnvironment(simulator)
    print("[SpeechManager] Skipping TTS on simulator (not supported)")
    return
    #endif
    // ... 原有邏輯
}
```

---

## 後續

- TTS 功能只能在實機上驗證
- 可向 WhisperKit repo 回報此 bug（NaN guard 缺失）
