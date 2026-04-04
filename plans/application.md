etLLM — Apple 平台 LLM Demo App
這是一個 iOS / macOS 原生 App，讓 LLM 能直接在裝置上推理（邊緣運算），核心使用 Meta 的 ExecuTorch 框架。

專案架構

apple/
├── etLLM.xcodeproj/          # Xcode 專案
├── Application/              # 主要 Swift 程式碼
│   ├── App.swift             # App 入口點
│   ├── ContentView.swift     # 主畫面（推理 UI）
│   ├── Message.swift         # 訊息資料模型
│   ├── MessageView.swift     # 訊息泡泡元件
│   ├── MessageListView.swift # 訊息列表
│   ├── ResourceManager.swift # 管理 model/tokenizer 路徑
│   ├── ResourceMonitor.swift # 記憶體/效能監控
│   ├── SpeechManager.swift   # TTS 語音播放
│   ├── LogManager.swift      # 推理 log 管理
│   ├── LogView.swift         # Log 檢視畫面
│   ├── ImagePicker.swift     # 圖片選取（多模態）
│   └── Constants.swift       # 各模型 Prompt 樣板
├── Entitlements/             # iOS/macOS 權限設定
└── Assets/                   # App Icon 等資源
支援的 LLM 模型
由 Message.swift:19-28 可知，目前支援 7 種模型：

模型	類型
LLaMA 3	純文字
Qwen 3	純文字（含 thinking）
Phi-4	純文字
Gemma 3	純文字
SmolLM 3	純文字
LLaVA	多模態（圖文）
Voxtral	語音理解
每個模型有對應的 Prompt 樣板定義在 Constants.swift。

核心元件說明
ResourceManager.swift

用 @AppStorage 持久化儲存 model .pte 和 tokenizer 路徑
驗證檔案是否存在
SpeechManager.swift

整合 TTSKit（Qwen3-TTS CoreML 模型，~1GB）
支援 loadModel() 非同步載入、speak() 播放、stop() 中斷
首次使用時自動從 HuggingFace 下載模型
ContentView.swift

使用 ExecuTorchLLM 的 TextRunner / MultimodalRunner 執行推理
包含圖片前處理（center crop、RGB normalize）供多模態模型使用
平台差異
功能	iOS	macOS
最低版本	iOS 17.0	macOS 14.0 (Sonoma, Apple Silicon)
圖片來源	相機 + 相簿	相簿（NSOpenPanel）
記憶體限制	需 increased-memory-limit entitlement	無需特殊設定
TTS	TTSKit + AVSpeech fallback	同左
技術棧
ExecuTorch — Meta 開源推理框架（.pte 模型格式）
TTSKit — argmaxinc/WhisperKit 衍生的 TTS 套件
SwiftUI — 全 UI
CoreML / XNNPACK / MPS — 後端加速選項