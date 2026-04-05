1. 沒有同步新程式：SpeechRecognitionManager 要加到 xcode 中
2. 

MicButtonView 目前只有 #if os(iOS)，macOS 也要能用，把條件移除：

3. 
This app has crashed because it attempted to access privacy-sensitive data without a usage description.  The app's Info.plist must contain an NSMicrophoneUsageDescription key with a string value explaining to the user how the app uses this dat