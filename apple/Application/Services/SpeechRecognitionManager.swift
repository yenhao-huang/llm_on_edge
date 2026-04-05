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
  @Published var isLiveMode = false

  var onTranscription: ((String) -> Void)?

  private var whisperKit: WhisperKit?
  private let modelName = "openai_whisper-base"
  private var recordingTask: Task<Void, Never>?
  private var silenceDetectionTask: Task<Void, Never>?

  // Silence detection config
  let silenceTimeout: Double = 5.0
  private let checkInterval: Double = 0.3
  // Samples used to calibrate baseline noise (~1s at 16kHz)
  private let calibrationSamples = 16000
  // Speech must be this many times louder than baseline to count
  private let speechRatio: Float = 1.5

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

  // MARK: - Live Mode

  func startLiveMode() {
    isLiveMode = true
    startLiveModeListening()
  }

  func stopLiveMode() {
    isLiveMode = false
    silenceDetectionTask?.cancel()
    silenceDetectionTask = nil
    if case .recording = recordingState {
      whisperKit?.audioProcessor.stopRecording()
      recordingTask?.cancel()
      recordingState = .readyToRecord
    }
    print("[STT] Live mode stopped")
  }

  /// Called by ContentView after generation (and optional TTS) completes to restart the listen cycle.
  func startLiveModeListening() {
    guard isLiveMode else { return }
    switch recordingState {
    case .readyToRecord:
      recordingTask = Task {
        await startRecording()
        await launchSilenceDetection()
      }
    case .idle:
      recordingTask = Task {
        await loadModelIfNeeded()
        guard isLiveMode, case .readyToRecord = recordingState else { return }
        await startRecording()
        await launchSilenceDetection()
      }
    default:
      break
    }
  }

  // MARK: - Private

  private func launchSilenceDetection() async {
    silenceDetectionTask?.cancel()
    let task = Task { await detectSilence() }
    silenceDetectionTask = task
    await task.value
  }

  private func detectSilence() async {
    var silenceDuration: Double = 0
    var hasSpeech = false
    var baselineRMS: Float = 0.01
    var calibrated = false

    print("[STT] Silence detection started (timeout: \(silenceTimeout)s)")

    while isLiveMode, case .recording = recordingState {
      try? await Task.sleep(nanoseconds: UInt64(checkInterval * 1_000_000_000))

      guard !Task.isCancelled, isLiveMode, case .recording = recordingState,
            let kit = whisperKit else { break }

      let samples = Array(kit.audioProcessor.audioSamples)

      // Calibration phase: use first ~1s to measure background noise
      guard samples.count >= calibrationSamples else {
        print("[STT] ⏳ buffering \(samples.count)/\(calibrationSamples) samples")
        continue
      }

      if !calibrated {
        let calWindow = Array(samples.prefix(calibrationSamples))
        baselineRMS = sqrt(calWindow.map { $0 * $0 }.reduce(0, +) / Float(calWindow.count))
        // Clamp baseline so a completely silent room still has reasonable threshold
        baselineRMS = max(baselineRMS, 0.005)
        print("[STT] 📐 Calibrated baseline noise rms=\(String(format: "%.4f", baselineRMS))  speech threshold=\(String(format: "%.4f", baselineRMS * speechRatio))")
        calibrated = true
      }

      let speechThreshold = baselineRMS * speechRatio

      // RMS of last checkInterval seconds
      let windowSize = min(Int(16000 * checkInterval), samples.count)
      let recent = Array(samples.suffix(windowSize))
      let rms = sqrt(recent.map { $0 * $0 }.reduce(0, +) / Float(recent.count))

      if rms > speechThreshold {
        hasSpeech = true
        silenceDuration = 0
        print("[STT] 🎙️ speech rms=\(String(format: "%.4f", rms))  threshold=\(String(format: "%.4f", speechThreshold))")
      } else {
        silenceDuration += checkInterval
        print("[STT] 🔇 silence rms=\(String(format: "%.4f", rms))  \(String(format: "%.1f", silenceDuration))s/\(silenceTimeout)s  hasSpeech=\(hasSpeech)")
      }

      if hasSpeech && silenceDuration >= silenceTimeout {
        print("[STT] ✅ Silence timeout reached — auto-transcribing")
        await stopAndTranscribe()
        break
      }
    }
  }

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
