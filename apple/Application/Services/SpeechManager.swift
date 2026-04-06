import Foundation
import TTSKit
import CosyVoiceTTS
import Qwen3TTS
import AudioCommon
import AVFoundation

// MARK: - Voice Clone Backend Type

enum VoiceCloneBackendType: String, CaseIterable {
  case cosyVoice = "CosyVoice3"
  case qwen3TTS  = "Qwen3-TTS"
}

// MARK: - Private helpers

private enum VoiceCloneError: Error {
  case modelNotLoaded
  case referenceNotReady
}

private protocol VoiceCloneBackend: AnyObject {
  var isLoaded: Bool { get }
  var isReferenceReady: Bool { get }
  var outputSampleRate: Int { get }
  func loadModel() async throws
  func setReferenceVoice(url: URL, referenceText: String) async throws
  func synthesize(text: String, language: String) throws -> [Float]
}

// MARK: - CosyVoice Backend

private final class CosyVoiceCloneBackend: VoiceCloneBackend {
  private var cosyModel: CosyVoiceTTSModel?
  private var camSpeaker: CamPlusPlusSpeaker?
  private var clonedEmbedding: [Float]?

  var isLoaded: Bool { cosyModel != nil && camSpeaker != nil }
  var isReferenceReady: Bool { clonedEmbedding != nil }
  var outputSampleRate: Int { 24000 }

  func loadModel() async throws {
    async let cosy = CosyVoiceTTSModel.fromPretrained()
    async let cam  = CamPlusPlusSpeaker.fromPretrained()
    (cosyModel, camSpeaker) = try await (cosy, cam)
  }

  func setReferenceVoice(url: URL, referenceText: String) async throws {
    guard let speaker = camSpeaker else { throw VoiceCloneError.modelNotLoaded }
    let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
    clonedEmbedding = try speaker.embed(audio: audio, sampleRate: 16000)
  }

  func synthesize(text: String, language: String) throws -> [Float] {
    guard let model = cosyModel, let embedding = clonedEmbedding else {
      throw VoiceCloneError.referenceNotReady
    }
    return model.synthesize(text: text, language: language, speakerEmbedding: embedding)
  }
}

// MARK: - Qwen3TTS Backend

private final class Qwen3TTSVoiceCloneBackend: VoiceCloneBackend {
  private var model: Qwen3TTSModel?
  private var encoder: SpeechTokenizerEncoder?
  private var referenceAudio: [Float]?
  private var storedReferenceText: String = ""

  var isLoaded: Bool { model != nil && encoder != nil }
  var isReferenceReady: Bool { referenceAudio != nil }
  var outputSampleRate: Int { 24000 }

  func loadModel() async throws {
    let (m, e) = try await Qwen3TTSModel.fromPretrainedWithEncoder()
    model = m
    encoder = e
  }

  func setReferenceVoice(url: URL, referenceText: String) async throws {
    referenceAudio = try AudioFileLoader.load(url: url, targetSampleRate: 24000)
    storedReferenceText = referenceText
  }

  func synthesize(text: String, language: String) throws -> [Float] {
    guard let model, let encoder, let refAudio = referenceAudio else {
      throw VoiceCloneError.referenceNotReady
    }
    return model.synthesizeWithVoiceCloneICL(
      text: text,
      referenceAudio: refAudio,
      referenceSampleRate: 24000,
      referenceText: storedReferenceText,
      language: language,
      codecEncoder: encoder
    )
  }
}

// MARK: - SpeechManager

@MainActor
final class SpeechManager: ObservableObject {
  @Published var isSpeaking = false
  @Published var isVoiceCloneLoaded = false

  var isEmbeddingReady: Bool { activeBackend?.isReferenceReady == true }

  private var tts: TTSKit?
  private var currentTask: Task<Void, Never>?
  private var activeBackend: (any VoiceCloneBackend)?

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

  func loadVoiceCloneModel(backend: VoiceCloneBackendType = .cosyVoice) async {
    let newBackend: any VoiceCloneBackend
    switch backend {
    case .cosyVoice:
      if activeBackend is CosyVoiceCloneBackend, activeBackend?.isLoaded == true { return }
      newBackend = CosyVoiceCloneBackend()
    case .qwen3TTS:
      if activeBackend is Qwen3TTSVoiceCloneBackend, activeBackend?.isLoaded == true { return }
      newBackend = Qwen3TTSVoiceCloneBackend()
    }
    print("[SpeechManager] Loading voice clone model (\(backend.rawValue))...")
    do {
      try await newBackend.loadModel()
      activeBackend = newBackend
      isVoiceCloneLoaded = true
      print("[SpeechManager] Voice clone model loaded successfully")
    } catch {
      print("[SpeechManager] Failed to load voice clone model: \(error)")
    }
  }

  func setReferenceVoice(url: URL, referenceText: String = "") async {
    guard let backend = activeBackend else {
      print("[SpeechManager] ⚠️ setReferenceVoice() skipped — backend not loaded")
      return
    }
    do {
      try await backend.setReferenceVoice(url: url, referenceText: referenceText)
      print("[SpeechManager] ✅ Speaker reference set")
    } catch {
      print("[SpeechManager] ❌ Failed to set reference voice: \(error)")
    }
  }

  func speakWithClonedVoice(_ text: String, language: String = "english") {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
      print("[SpeechManager] ⚠️ speakWithClonedVoice() skipped — empty text")
      return
    }
    guard let backend = activeBackend, backend.isReferenceReady else {
      print("[SpeechManager] ⚠️ Voice clone not ready — falling back to TTSKit")
      speak(trimmed)
      return
    }

    currentTask?.cancel()
    currentTask = Task {
      isSpeaking = true
      defer { isSpeaking = false }
      do {
        let samples = try backend.synthesize(text: trimmed, language: language)
        guard !samples.isEmpty, !Task.isCancelled else { return }
        try await playPCM(samples: samples, sampleRate: backend.outputSampleRate)
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
