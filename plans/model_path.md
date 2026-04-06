  ---                                                                   
  WhisperKit（STT）                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                             
  管理位置: SpeechRecognitionManager.swift:24                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                             
  private let modelName = "openai_whisper-large-v3"               
                                                                                                                                                                                                                                                                             
  - Model name 寫死在 SpeechRecognitionManager 的 private 屬性                                                                                                                                                                                                               
  - 路徑由 WhisperKit 自行管理，下載後快取至：
  ~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/openai_whisper-large-v3/                                                                                                                                                                                        
  - 透過 WhisperKitConfig(model: modelName, download: true) 傳入，TTSKit 處理下載與快取                                                                                                                                                                                      
                                                                                                                                                                                                                                                                             
  ---                                                                                                                                                                                                                                                                        
  TTSKit（TTS）                                                                                                                                                                                                                                                              
                                                                  
  管理位置: SpeechManager.swift:24
                                                                                                                                                                                                                                                                             
  tts = try await TTSKit(TTSKitConfig(model: .qwen3TTS_0_6b))
                                                                                                                                                                                                                                                                             
  - 以 enum TTSKitConfig.Model.qwen3TTS_0_6b 選擇 model（TTSKit 是 closed binary package）                                                                                                                                                                                   
  - TTSKit 內部封裝 Qwen3TTSCoreMLModel，實際呼叫：                                                                                                                                                                                                                          
  HuggingFaceDownloader.getCacheDirectory(for: "aufklarer/Qwen3-TTS-CoreML")                                                                                                                                                                                                 
  - 快取路徑：                                                                                                                                                                                                                                                               
  ~/Library/Caches/qwen3-speech/models/aufklarer/Qwen3-TTS-CoreML/                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                             
  ---                                                                                                                                                                                                                                                                        
  speech-swift（CosyVoice3 + CAM++）
                                                                                                                                                                                                                                                                             
  管理位置: SpeechManager.swift:38-39                             

  async let cosy = CosyVoiceTTSModel.fromPretrained()
  async let cam  = CamPlusPlusSpeaker.fromPretrained()                                                                                                                                                                                                                       
  
  model ID 的預設值各自定義在：                                                                                                                                                                                                                                              
                                                                  
  ┌────────────────────┬─────────────────────────────┬──────────────────────────────────────┐                                                                                                                                                                                
  │       Model        │          定義位置           │               預設 ID                │
  ├────────────────────┼─────────────────────────────┼──────────────────────────────────────┤                                                                                                                                                                                
  │ CosyVoiceTTSModel  │ CosyVoiceTTS.swift:57       │ aufklarer/CosyVoice3-0.5B-MLX-4bit   │
  ├────────────────────┼─────────────────────────────┼──────────────────────────────────────┤
  │ CamPlusPlusSpeaker │ CamPlusPlusSpeaker.swift:18 │ aufklarer/CamPlusPlus-Speaker-CoreML │                                                                                                                                                                                
  └────────────────────┴─────────────────────────────┴──────────────────────────────────────┘                                                                                                                                                                                
                                                                                                                                                                                                                                                                             
  兩者都透過 HuggingFaceDownloader.getCacheDirectory() 統一管理，快取至：                                                                                                                                                                                                    
  ~/Library/Caches/qwen3-speech/models/<org>/<model-name>/        
                                                                                                                                                                                                                                                                             
  快取根目錄可用環境變數覆蓋（HuggingFaceDownloader.swift:178）：                                                                                                                                                                                                            
  QWEN3_CACHE_DIR=<path>  # 優先
  QWEN3_ASR_CACHE_DIR=<path>  # 舊版 fallback                                