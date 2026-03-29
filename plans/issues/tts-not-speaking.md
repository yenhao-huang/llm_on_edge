# Issue: TTS 生成後沒有發話

**狀態：已解決**
**解決日期：2026-03-29**

---

## 症狀
LLM 成功生成文字（顯示在 UI），SpeechManager 沒有發話，無任何錯誤訊息。

---

## 完整 Debug 過程

### Step 1：排除 TTSKit pipeline 本身的問題

```bash
cd ~/Library/Developer/Xcode/DerivedData/etLLM-.../SourcePackages/checkouts/whisperkit
swift run whisperkit-cli tts --text "Hello, TTSKit works!" --play
```

輸出 `output.m4a`（53KB，2.48 秒）。**TTSKit CLI 正常**。

### Step 2：確認 CoreML 模型已下載

```bash
find ~/Library/Containers/org.pytorch.executorch.etllm.macos/ -name "*.bin"
```

三個模型都在：MultiCodeEmbedder / SpeechDecoder / CodeDecoder。**不是模型缺失問題**。

### Step 3：捕捉 App stdout

```bash
"$APP_PATH" > /tmp/etllm_stdout.txt 2>&1 &
```

LLM 生成完成後觀察 `/tmp/etllm_stdout.txt`：**完全沒有任何 `[SpeechManager]` 輸出**。

這代表 `loadModel()` 沒有被叫到，或根本跑的不是含 TTSKit 的新 binary。

### Step 4：加 debug prints，重新編譯

```swift
// SpeechManager.swift
func loadModel() async {
    print("[SpeechManager] Loading TTSKit model...")
    // 成功: print("[SpeechManager] TTSKit loaded successfully")
    // 失敗: print("[SpeechManager] Failed to load TTSKit: \(error)")
}

func speak(_ text: String) {
    print("[SpeechManager] speak() called, tts=\(tts != nil ? "loaded" : "nil"), text=\(trimmed.prefix(50))")
    // tts.play() 前後也加 print
}
```

```bash
cd apple/
xcodebuild -project etLLM.xcodeproj -scheme etLLM-macOS -configuration Debug \
  CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO build
# → BUILD SUCCEEDED
```

### Step 5：Kill 舊 process，強制跑新 binary ← **根本解法**

```bash
kill 918   # 舊 process（9:21PM 啟動，早於本次 code 修改）

# 直接執行 binary，不用 open（open 會複用同 bundle ID 的既有 process）
"/path/to/DerivedData/.../etLLM-macOS.app/Contents/MacOS/etLLM-macOS" \
  > /tmp/etllm_stdout.txt 2>&1 &
# → App PID: 5824
```

操作 App（load model + tokenizer + 輸入 prompt），stdout 開始出現 LLM 輸出，**TTS 正常發話**。

---

## 根本原因：跑的是舊 binary

整個問題**不是 code bug，是部署問題**。

```
Code 改了 → 重新編譯 ✓ → 用 `open App.app` 重啟
                              ↓
              macOS: 偵測到同 bundle ID 的 app 已在跑 (PID 918)
                              ↓
              直接把舊 process 帶到前景，不換 binary
                              ↓
              跑的還是 TTSKit 整合前的舊版 SpeechManager
                              ↓
              完全靜默（speak() 使用 AVSpeechSynthesizer，
              但新 ContentView 呼叫方式不同）
```

---

## 解法

1. **Kill 舊 process**
   ```bash
   kill $(ps aux | grep etLLM-macOS | grep -v grep | awk '{print $2}')
   ```

2. **重新編譯**（清除舊 binary）
   ```bash
   xcodebuild -project etLLM.xcodeproj -scheme etLLM-macOS \
     -configuration Debug CODE_SIGNING_ALLOWED=NO build
   ```

3. **直接執行新 binary 或用 Xcode ▶ Run**

   Xcode Run 會自動 kill + 重啟，是最安全的方式。

---

## 後續：Xcode Run 出現新 crash（tiktoken.h:46）

**症狀：** 用 Xcode ▶ Run 啟動後，執行到 tokenizer 載入時 crash：

```
0x105e19e10 <+236>: bl     0x106745eb4   ; symbol stub for: abort
->  0x105e19e14 <+240>: b      0x105e19e18   ; at tiktoken.h:46:7
```

**初步判斷：**

| 可能原因 | 說明 |
|---|---|
| Tokenizer 格式不符 | `tokenizer.json` 是 HuggingFace JSON 格式，ExecuTorch tiktoken 期待 `.model` 或 `.tiktoken` binary 格式 |
| DerivedData cache 問題 | Xcode 使用 `executorch_llm_debug`（assertion 全開），stale cache 可能導致 framework mismatch |
| ET_CHECK 斷言失敗 | Debug build 的 `ET_CHECK_MSG` 在條件不滿足時呼叫 `abort()`，release build 不觸發 |

**排查方向：**
1. Product → Clean Build Folder（Shift+Cmd+K）清 DerivedData
2. 重新編譯後用 Xcode Run
3. 確認選取的 tokenizer 格式（qwen3 可能需要 `.tiktoken` 或 `.bin` 格式，不是 JSON）

---

## 教訓

| 情況 | 行為 |
|---|---|
| `open App.app`（app 已在跑） | macOS 喚醒舊 process，**不換 binary** |
| `open App.app`（app 未在跑） | 正常啟動新 binary |
| 直接執行 binary 路徑 | 強制啟動新 process，永遠用最新 binary |
| Xcode ▶ Run | Kill 舊 process 再重啟，永遠用最新 binary，是開發時最可靠的方式 |

**遇到「改了 code 但行為沒變」，第一件事：確認跑的是不是新 binary。**
