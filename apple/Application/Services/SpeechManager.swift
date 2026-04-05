import Foundation
import TTSKit

@MainActor
final class SpeechManager: ObservableObject {
  @Published var isSpeaking = false

  private var tts: TTSKit?
  private var currentTask: Task<Void, Never>?

  func loadModel() async {
    guard tts == nil else { return }
    print("[SpeechManager] Loading TTSKit model...")
    do {
      tts = try await TTSKit(TTSKitConfig(model: .qwen3TTS_0_6b))
      print("[SpeechManager] TTSKit loaded successfully")
    } catch {
      print("[SpeechManager] Failed to load TTSKit: \(error)")
    }
  }

  func speak(_ text: String) {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

    // Diagnostic: model state
    guard let tts else {
      print("[SpeechManager] ⚠️ speak() skipped — model not loaded (call loadModel() first)")
      return
    }
    guard !trimmed.isEmpty else {
      print("[SpeechManager] ⚠️ speak() skipped — empty text")
      return
    }

    print("[SpeechManager] ✅ model loaded, speaking \(trimmed.count) chars: \"\(trimmed.prefix(80))\"")

    currentTask?.cancel()
    currentTask = Task {
      isSpeaking = true
      defer { isSpeaking = false }
      do {
        try await tts.play(text: trimmed, playbackStrategy: .auto)
        print("[SpeechManager] ✅ tts.play() completed")
      } catch {
        if !(error is CancellationError) {
          print("[SpeechManager] ❌ TTS error: \(error)")
        }
      }
    }
  }

  func stop() {
    currentTask?.cancel()
    currentTask = nil
  }
}
