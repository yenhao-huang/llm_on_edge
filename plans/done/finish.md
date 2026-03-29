⏺ 完成。以下是所有改動的摘要：                  

  新增檔案：                                                                                                                                                                                                                                                                 
  - apple/Application/SpeechManager.swift — 從 git 恢復，使用 TTSKit 播放語音
                                                                                                                                                                                                                                                                             
  修改：                                                                                                                                                                                                                                                                   
  - MessageView.swift — 接受 speechManager 參數，assistant 訊息的 context menu 新增 Speak / Stop Speaking                                                                                                                                                                    
  - MessageListView.swift — 接受並往下傳遞 speechManager                                                                                                                                                                                                                     
  - ContentView.swift:                                  
    - 新增 @StateObject private var speechManager = SpeechManager() 及 @State private var autoSpeak = false                                                                                                                                                                  
    - iOS 工具列右邊加入喇叭 toggle 按鈕（藍色 = 開啟）                                                                                                                                                                                                                    
    - macOS sidebar 加入 Auto Speak 按鈕                                                                                                                                                                                                                                     
    - onAppear 時在背景載入 TTS 模型                                                                                                                                                                                                                                         
    - 生成結束後若 autoSpeak 為 true，自動對最後一則 assistant 訊息呼叫 speechManager.speak()                                                                                                                                                                                
  - etLLM.xcodeproj/project.pbxproj — 加入 SpeechManager.swift 到 iOS 及 macOS 兩個 target 的 Sources build phase                                                                                                                                                            
                                                                                                                       


  1 . 開 Xcode，開啟 apple/etLLM.xcodeproj           
  2. File → Add Package Dependencies...                                                                                                                                                                                                                                      
  3. 輸入：https://github.com/argmaxinc/WhisperKit                                                                                                                                                                                                                           
  4. 版本選 Up to Next Major Version（或指定你用過的版本）                                                                                                                                                                                                                   
  5. 只勾選 TTSKit（不需要 WhisperKit、SpeakerKit）                                                                                                                                                                                                                          
  6. Add to 兩個 target：etLLM（iOS）和 etLLM-macOS   

    ---                                                                                                                                                                                                                                                                        
  如果你當時只加了一個 target，事後補加：                         
                                                                                                                                                                                                                                                                             
  1. 點左側 Project navigator 最上層 etLLM (藍色 icon)
  2. 選 target（例如 etLLM-macOS）                                                                                                                                                                                                                                           
  3. 上方 tab 選 General                                                                                                                                                                                                                                                     
  4. 往下滑到 Frameworks, Libraries, and Embedded Content                                                                                                                                                                                                                    
  5. 點 + → 找 TTSKit → Add    


  ⏺ Update(~/Library/Developer/Xcode/DerivedData/etLLM-cockdxsccottsfgplcxmhforgaka/SourcePackages/checkouts/whisperkit/Sources/TTSKit/Utilities/Sampling.swift)                                                                                                               
  ⎿  Added 4 lines, removed 5 lines                                                                                                                                                                                                                                        
      236              let probsArray = await topKProbs.toFloatArray()                                                                                                                                                                                                       
      237              let idxArray = await topKIndices.toIntArray()
      238              let probSum = probsArray.reduce(0, +)                                                                                                                                                                                                                 
      239 -            // Patch: probSum == 0 means all top-k logits are NaN/zero after softmax.                                                                                                                                                                           
      240 -            // Treat as uniform over top-k rather than crashing with Float.random(in: 0..<0).                                                                                                                                                              
      241 -            guard probSum > 0 else {                                                                                                                                                                                                                       
      242 -                print("[TTSKit] ⚠️  sampleFromProbs: probSum == 0, falling back to last top-k token")                                                                                                                                                       
      243 -                return idxArray.last.map(Int32.init) ?? Int32(vocabSize - 1)                                                                                                                                                                               
      239 +            // Patch: guard against probSum == 0, NaN, or Inf — all cause Float.random to crash.                                                                                                                                                           
      240 +            guard probSum > 0, probSum.isFinite else {                                                                                                                                                                                                     
      241 +                print("[TTSKit] ⚠️  sampleFromProbs: probSum=\(probSum), falling back to top-1 token")                                                                                                                                                      
      242 +                return idxArray.first.map(Int32.init) ?? Int32(vocabSize - 1)                                                                                                                                                                              
      243              }
      244              let randomValue = Float.random(in: 0..<probSum, using: &rng)
      245              var cumulativeSum: Float = 0

