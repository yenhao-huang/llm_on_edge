# Issue: Xcode Run 時 tiktoken.h:46 abort() crash

**狀態：已修復**
**修復日期：2026-03-29**

---

## 症狀

用 Xcode ▶ Run 啟動 App 後，載入模型時 crash：

```
0x105e19e08 <+228>: tbnz   w8, #0x0, 0x105e19e50     ; <+300> at tiktoken.h
0x105e19e0c <+232>: b      0x105e19e10               ; <+236> at tiktoken.h:46:7
0x105e19e10 <+236>: bl     0x106745eb4               ; symbol stub for: abort
->  0x105e19e14 <+240>: b      0x105e19e18               ; <+244> at tiktoken.h:46:7
```

CLI build（`CODE_SIGNING_ALLOWED=NO`）正常，只有 Xcode signed build 才 crash。

---

## 調查過程

### Step 1：確認 tiktoken.h:46 做什麼

找到 tiktoken.h 內容（來自 `pytorch_tokenizers` 套件）：

```cpp
// tiktoken.h:43-48
_bos_token_index(bos_token_index),
_eos_token_index(eos_token_index) {
  if (_bos_token_index >= _special_tokens->size() ||
      _eos_token_index >= _special_tokens->size()) {
    abort();  // ← line 46
  }
}
```

**abort 條件**：`bos_token_index(0)` 或 `eos_token_index(1)` >= `special_tokens.size()`。

當 `special_tokens` 是空 vector 時（size=0），`0 >= 0` 為 true → abort。

### Step 2：確認 qwen3 沒有傳 special tokens

```swift
// ContentView.swift:1145-1149
case .qwen3, .phi4, .smollm3:
    runnerHolder.textRunner = runnerHolder.textRunner ?? TextRunner(
        modelPath: modelPath,
        tokenizerPath: tokenizerPath
        // ← 沒有 specialTokens，會用空 array 呼叫 designated init
    )
```

BUT：Llama 有傳 special tokens（11 個），qwen3 沒有。但 qwen3 用的是 HuggingFace JSON tokenizer，理論上不應該走到 Tiktoken 的 constructor。

### Step 3：確認 tokenizer 路徑相同

兩個 UserDefaults（sandboxed + non-sandboxed）都指向：
- `modelPath`: `qwen3-0.6b.pte` ✓
- `tokenizerPath`: `tokenizer.json` ✓

路徑一樣，但 CLI 正常、Xcode crash。

### Step 4：發現 App Sandbox 差異 ← 根本原因

查 `AppMac.entitlements`：

```xml
<key>com.apple.security.app-sandbox</key>
<true/>
<key>com.apple.security.files.user-selected.read-write</key>
<true/>
```

`com.apple.security.files.user-selected.read-write` 只允許存取**當次 session 透過 file picker 選取**的檔案。

`ResourceManager` 只存路徑字串（`@AppStorage`），**沒有存 security-scoped bookmark**。

```
App 啟動 → 讀 UserDefaults 拿到路徑 → 嘗試開檔
                                          ↓
              Sandbox: 這個路徑不在本次 session 的允許清單 → 拒絕
                                          ↓
              tokenizer file open 失敗（無 error，靜默失敗）
                                          ↓
              ExecuTorch tokenizer 嘗試 fallback → Tiktoken
                                          ↓
              Tiktoken constructor: special_tokens 為空 → abort()
```

**CLI build（CODE_SIGNING_ALLOWED=NO）**：無 sandbox，可以直接讀任意路徑，正常。

---

## 修法

### ResourceManager.swift：加 security-scoped bookmark 支援

新增兩個方法：

```swift
// 選檔案時呼叫：儲存 security-scoped bookmark
func saveBookmark(for url: URL, isModel: Bool) {
    guard let bookmark = try? url.bookmarkData(
        options: .withSecurityScope, ...
    ) else { return }
    UserDefaults.standard.set(bookmark, forKey: key)
}

// App 啟動時呼叫：恢復 sandbox 存取權
func restoreAccess() {
    // 讀 bookmark → URL.resolvingBookmarkData → startAccessingSecurityScopedResource()
}
```

### ContentView.swift：兩處修改

**1. 選檔案時啟動 security-scoped access + 存 bookmark**（`handleFileImportResult`）：
```swift
_ = url.startAccessingSecurityScopedResource()  // ← 新增：啟動當次 session 存取權
switch pickerType {
case .model:
    resourceManager.modelPath = url.path
    resourceManager.saveBookmark(for: url, isModel: true)   // ← 新增：存跨 session bookmark
case .tokenizer:
    resourceManager.tokenizerPath = url.path
    resourceManager.saveBookmark(for: url, isModel: false)  // ← 新增
}
```

`startAccessingSecurityScopedResource()` 必須在這裡呼叫，不能只靠 bookmark。原因：
- Bookmark 只解決**跨 session**（下次啟動）的存取問題
- `runnerQueue.async` 的 file open 發生在**當次 session**，需要先 start access

**2. App 啟動時 restore**（macOS + iOS 各一個 `onAppear`）：
```swift
.onAppear {
    resourceManager.restoreAccess()  // ← 新增，必須在載入 model 前
    Task { await speechManager.loadModel() }
    ...
}
```

---

## 為什麼 CLI 不 crash，Xcode 才 crash

| | CLI build（CODE_SIGNING_ALLOWED=NO） | Xcode build（signed）|
|---|---|---|
| App Sandbox | 不強制（無有效 code sign） | 強制（entitlements 生效） |
| 可讀任意路徑 | ✓ | ✗（需 user-selected 或 bookmark）|
| tokenizer.json 讀取 | 成功 → HF JSON tokenizer | 失敗 → fallback → Tiktoken crash |

---

## 教訓

- **macOS sandbox + `user-selected.read-write`** 的存取權只在當次 session 有效
- 跨 session 存取使用者選取的檔案必須使用 **security-scoped bookmark**
- ExecuTorch tokenizer 的 fallback 順序在檔案讀取失敗時會走到 Tiktoken，而 Tiktoken constructor 遇到空 special_tokens 直接 `abort()` 而不是回傳 error，這是 ExecuTorch 的 debug-only 行為（release build 可能不 crash）
