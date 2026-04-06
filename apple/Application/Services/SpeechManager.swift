import Foundation
import TTSKit
import CosyVoiceTTS
import AudioCommon
import AVFoundation

@MainActor
final class SpeechManager: ObservableObject {
  @Published var isSpeaking = false
  @Published var isVoiceCloneLoaded = false

  var isEmbeddingReady: Bool { clonedEmbedding != nil }

  private var tts: TTSKit?
  private var currentTask: Task<Void, Never>?

  // Voice clone state
  private var cosyModel: CosyVoiceTTSModel?
  private var camSpeaker: CamPlusPlusSpeaker?
  private var clonedEmbedding: [Float]?

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

  // MARK: - Voice Clone

  /// Load CosyVoice3 + CAM++ speaker encoder for voice cloning.
  func loadVoiceCloneModel() async {
    guard cosyModel == nil else { return }
    print("[SpeechManager] Loading CosyVoice3 + CAM++ models...")
    do {
      async let cosy = CosyVoiceTTSModel.fromPretrained()
      async let cam = CamPlusPlusSpeaker.fromPretrained()
      (cosyModel, camSpeaker) = try await (cosy, cam)
      print("[SpeechManager] Voice clone models loaded successfully")
      isVoiceCloneLoaded = true
    } catch {
      print("[SpeechManager] Failed to load voice clone models: \(error)")
    }
  }

  /// Set the reference audio URL for voice cloning.
  /// Extracts a 192-dim CAM++ speaker embedding from the audio file.
  func setReferenceVoice(url: URL) async {
    guard let speaker = camSpeaker else {
      print("[SpeechManager] ⚠️ setReferenceVoice() skipped — CAM++ model not loaded")
      return
    }
    do {
      let refAudio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
      clonedEmbedding = try speaker.embed(audio: refAudio, sampleRate: 16000)
      print("[SpeechManager] ✅ Speaker embedding extracted (\(clonedEmbedding?.count ?? 0) dims)")
    } catch {
      print("[SpeechManager] ❌ Failed to extract speaker embedding: \(error)")
    }
  }

  /// Speak using cloned voice if embedding is available, otherwise fall back to TTSKit.
  func speakWithClonedVoice(_ text: String, language: String = "english") {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
      print("[SpeechManager] ⚠️ speakWithClonedVoice() skipped — empty text")
      return
    }

    guard let model = cosyModel, let embedding = clonedEmbedding else {
      print("[SpeechManager] ⚠️ Voice clone not ready — falling back to TTSKit")
      speak(trimmed)
      return
    }

    currentTask?.cancel()
    currentTask = Task {
      isSpeaking = true
      defer { isSpeaking = false }
      let samples = model.synthesize(
        text: trimmed,
        language: language,
        speakerEmbedding: embedding
      )
      guard !samples.isEmpty, !Task.isCancelled else { return }
      do {
        try await playPCM(samples: samples, sampleRate: 24000)
      } catch {
        if !(error is CancellationError) {
          print("[SpeechManager] ❌ Voice clone playback error: \(error)")
        }
      }
    }
  }

  // MARK: - TTSKit

  func speak(_ text: String) {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

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

  // MARK: - Private

  private func playPCM(samples: [Float], sampleRate: Int) async throws {
    let engine = AVAudioEngine()
    let playerNode = AVAudioPlayerNode()
    engine.attach(playerNode)

    guard let format = AVAudioFormat(
      commonFormat: .pcmFormatFloat32,
      sampleRate: Double(sampleRate),
      channels: 1,
      interleaved: false
    ) else { return }

    engine.connect(playerNode, to: engine.mainMixerNode, format: format)
    try engine.start()

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
    buffer.frameLength = AVAudioFrameCount(samples.count)
    samples.withUnsafeBufferPointer { ptr in
      buffer.floatChannelData?[0].update(from: ptr.baseAddress!, count: samples.count)
    }

    return try await withCheckedThrowingContinuation { continuation in
      playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
        continuation.resume()
      }
      playerNode.play()
    }
  }
}
