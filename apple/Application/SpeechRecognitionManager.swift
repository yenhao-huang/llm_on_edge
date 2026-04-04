import Foundation
import WhisperKit

@MainActor
final class SpeechRecognitionManager: ObservableObject {

  enum RecordingState: Equatable {
    case idle
    case loadingModel(progress: Double)
    case readyToRecord
    case requestingPermission
    case permissionDenied
    case recording
    case transcribing
    case error(String)
  }

  @Published var recordingState: RecordingState = .idle

  var onTranscription: ((String) -> Void)?

  private var whisperKit: WhisperKit?
  private let modelName = "openai_whisper-base"
  private var recordingTask: Task<Void, Never>?

  // MARK: - Public API

  func loadModelIfNeeded() async {
    guard whisperKit == nil else { return }
    print("[STT] Loading WhisperKit model...")
    recordingState = .loadingModel(progress: 0.0)
    do {
      let config = WhisperKitConfig(
        model: modelName,
        verbose: false,
        logLevel: .error,
        prewarm: false,
        load: true,
        download: true
      )
      let kit = try await WhisperKit(config)
      whisperKit = kit
      recordingState = .readyToRecord
      print("[STT] WhisperKit loaded successfully")
    } catch {
      print("[STT] Failed to load WhisperKit: \(error)")
      recordingState = .error("Failed to load STT model: \(error.localizedDescription)")
    }
  }

  func toggleRecording() {
    switch recordingState {
    case .recording:
      recordingTask = Task { await stopAndTranscribe() }
    case .readyToRecord:
      recordingTask = Task { await startRecording() }
    case .idle:
      recordingTask = Task {
        await loadModelIfNeeded()
        if case .readyToRecord = recordingState {
          await startRecording()
        }
      }
    default:
      break
    }
  }

  func stopIfRecording() async {
    guard case .recording = recordingState else { return }
    await stopAndTranscribe()
  }

  func unloadModel() async {
    guard case .recording = recordingState else {
      await whisperKit?.unloadModels()
      whisperKit = nil
      recordingState = .idle
      return
    }
    whisperKit?.audioProcessor.stopRecording()
    recordingTask?.cancel()
    await whisperKit?.unloadModels()
    whisperKit = nil
    recordingState = .idle
  }

  // MARK: - Private

  private func startRecording() async {
    guard let kit = whisperKit else { return }
    let granted = await requestPermission()
    guard granted else {
      recordingState = .permissionDenied
      print("[STT] Microphone permission denied")
      return
    }
    do {
      try kit.audioProcessor.startRecordingLive(callback: nil)
      recordingState = .recording
      print("[STT] Recording started")
    } catch {
      print("[STT] Failed to start recording: \(error)")
      recordingState = .error("Recording failed: \(error.localizedDescription)")
    }
  }

  private func stopAndTranscribe() async {
    guard let kit = whisperKit else { return }
    recordingState = .transcribing
    print("[STT] Stopping recording, transcribing...")

    let audioSamples = Array(kit.audioProcessor.audioSamples)
    kit.audioProcessor.stopRecording()

    guard !audioSamples.isEmpty else {
      print("[STT] No audio samples captured")
      recordingState = .readyToRecord
      return
    }

    do {
      let results = try await kit.transcribe(audioArray: audioSamples)
      let text = results.map(\.text).joined()
        .trimmingCharacters(in: .whitespacesAndNewlines)
      print("[STT] Transcribed: \"\(text)\"")
      if !text.isEmpty {
        onTranscription?(text)
      }
    } catch {
      print("[STT] Transcription failed: \(error)")
      recordingState = .error("Transcription failed: \(error.localizedDescription)")
      return
    }
    recordingState = .readyToRecord
  }

  private func requestPermission() async -> Bool {
    recordingState = .requestingPermission
    return await AudioProcessor.requestRecordPermission()
  }
}
