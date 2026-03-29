# etLLM macOS App 開啟方式

## 用 xcodebuild + open（推薦，不需 signing）

```bash
cd /Users/yenhaohuang/Desktop/side_project/edge_devices/llm-on-iphone/apple

# Build
xcodebuild -project etLLM.xcodeproj -scheme "etLLM-macOS" -configuration Debug \
  CODE_SIGN_IDENTITY="-" \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGNING_ALLOWED=NO \
  build

# Launch
open ~/Library/Developer/Xcode/DerivedData/etLLM-cockdxsccottsfgplcxmhforgaka/Build/Products/Debug/etLLM-macOS.app
```

## 用 Xcode GUI (未成功)

1. 開啟 `apple/etLLM.xcodeproj`
2. Scheme 切換成 **etLLM-macOS** → destination **My Mac**
3. Signing & Capabilities → 選 Development Team
4. **Cmd+R**

## 注意

- TTSKit 首次啟動會從 HuggingFace 下載 Qwen3-TTS CoreML 模型（~1GB），需等幾分鐘
- Speak 測試：送訊息 → 等 LLM 回覆 → **右鍵 assistant 訊息 → Speak**
- iOS Simulator 不支援 TTSKit（無 ANE），必須用 macOS 或真機測試
