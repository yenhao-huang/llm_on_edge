/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Combine
import ExecuTorchLLM
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Picker Type

enum PickerType {
  case model
  case tokenizer
}

// MARK: - ChatViewModel

@MainActor
class ChatViewModel: ObservableObject {
  @Published var messages: [Message] = []
  @Published var isGenerating = false
  @Published var shouldStopGenerating = false
  @Published var prompt = ""
  @Published var thinkingMode = false
  @Published var autoSpeak = false
  @Published var voiceCloneEnabled = false
  @Published var isLoadingVoiceClone = false
  @Published var voiceCloneBackendType: VoiceCloneBackendType = .qwen3TTS
  @Published var voiceCloneReferenceText: String = ""
  @Published var selectedImage: PlatformImage?

  let resourceManager = ResourceManager()
  let resourceMonitor = ResourceMonitor()
  let logManager = LogManager()
  let speechManager: SpeechManager
  let speechRecognitionManager: SpeechRecognitionManager
  let llmService = LLMGenerationService()

  var lastPreloadedKey: String?
  private var cancellables = Set<AnyCancellable>()

  init() {
    speechManager = SpeechManager()
    speechRecognitionManager = SpeechRecognitionManager()
    setupObservation()
  }

  private func setupObservation() {
    // Forward nested ObservableObject changes so the View updates correctly
    speechManager.objectWillChange
      .sink { [weak self] _ in self?.objectWillChange.send() }
      .store(in: &cancellables)
    speechRecognitionManager.objectWillChange
      .sink { [weak self] _ in self?.objectWillChange.send() }
      .store(in: &cancellables)
    resourceManager.objectWillChange
      .sink { [weak self] _ in self?.objectWillChange.send() }
      .store(in: &cancellables)
    resourceMonitor.objectWillChange
      .sink { [weak self] _ in self?.objectWillChange.send() }
      .store(in: &cancellables)
    logManager.objectWillChange
      .sink { [weak self] _ in self?.objectWillChange.send() }
      .store(in: &cancellables)

    // Live conversation restart logic
    $isGenerating
      .dropFirst()
      .sink { [weak self] isGenerating in
        guard let self, !isGenerating,
              speechRecognitionManager.isLiveMode, !autoSpeak else { return }
        Task {
          try? await Task.sleep(nanoseconds: 500_000_000)
          self.speechRecognitionManager.startLiveModeListening()
        }
      }
      .store(in: &cancellables)

    speechManager.$isSpeaking
      .dropFirst()
      .sink { [weak self] isSpeaking in
        guard let self, !isSpeaking,
              speechRecognitionManager.isLiveMode, !isGenerating else { return }
        Task {
          try? await Task.sleep(nanoseconds: 500_000_000)
          self.speechRecognitionManager.startLiveModeListening()
        }
      }
      .store(in: &cancellables)
  }

  var isInputEnabled: Bool { resourceManager.isModelValid && resourceManager.isTokenizerValid }

  // MARK: - Setup

  func setup() {
    do {
      try resourceManager.createDirectoriesIfNeeded()
    } catch {
      withAnimation {
        messages.append(Message(type: .info, text: "Error creating content directories: \(error.localizedDescription)"))
      }
    }
    Task { await speechManager.loadModel() }
    Task { await speechRecognitionManager.loadModelIfNeeded() }
  }

  // MARK: - Generate

  func generate() {
    guard !prompt.isEmpty else { return }
    guard speechRecognitionManager.recordingState != .recording else { return }
    isGenerating = true
    shouldStopGenerating = false
    let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    let modelPath = resourceManager.modelPath
    let tokenizerPath = resourceManager.tokenizerPath
    let modelType = ModelType.fromPath(modelPath)
    let capturedImage = selectedImage
    prompt = ""
    messages.append(Message(text: text))
    let outputType = outputMessageType(for: modelType)
    messages.append(Message(type: outputType))

    llmService.runnerQueue.async { [weak self] in
      guard let self else { return }

      defer {
        DispatchQueue.main.async {
          self.isGenerating = false
          self.selectedImage = nil
          if self.autoSpeak, let last = self.messages.last, !last.text.isEmpty {
            switch last.type {
            case .llamagenerated, .llavagenerated, .qwengenerated, .qwen3_5generated,
                 .phi4generated, .gemma3generated, .smollm3generated, .voxtralgenerated:
              if self.voiceCloneEnabled {
                self.speechManager.speakWithClonedVoice(last.text)
              } else {
                self.speechManager.speak(last.text)
              }
            default: break
            }
          }
        }
      }

      // Load model
      do {
        if let duration = try self.llmService.loadModelIfNeededSync(
          modelPath: modelPath, tokenizerPath: tokenizerPath
        ) {
          DispatchQueue.main.async {
            withAnimation {
              var msg = self.messages.removeLast()
              msg.type = .info
              msg.text = "Model loaded in \(String(format: "%.2f", duration)) s"
              self.messages.append(msg)
              self.messages.append(Message(type: outputType))
            }
          }
        }
      } catch {
        DispatchQueue.main.async {
          withAnimation {
            var msg = self.messages.removeLast()
            msg.type = .info
            msg.text = "Model loading failed: error \((error as NSError).code)"
            self.messages.append(msg)
          }
        }
        return
      }

      guard !self.shouldStopGenerating else {
        DispatchQueue.main.async { withAnimation { _ = self.messages.removeLast() } }
        return
      }

      defer {
        self.llmService.runnerHolder.textRunner?.reset()
        self.llmService.runnerHolder.multimodalRunner?.reset()
      }

      do {
        try self.llmService.generate(
          text: text,
          modelPath: modelPath,
          thinkingMode: self.thinkingMode,
          image: capturedImage,
          onEvent: { event in
            DispatchQueue.main.async { self.handleGenerationEvent(event) }
          },
          shouldStop: { self.shouldStopGenerating }
        )
      } catch let err as LLMGenerationError {
        DispatchQueue.main.async {
          withAnimation {
            var msg = self.messages.removeLast()
            msg.type = .info
            msg.text = err.errorDescription ?? "Generation error"
            self.messages.append(msg)
          }
        }
      } catch {
        DispatchQueue.main.async {
          withAnimation {
            var msg = self.messages.removeLast()
            msg.type = .info
            msg.text = "Text generation failed: error \((error as NSError).code)"
            self.messages.append(msg)
          }
        }
      }
    }
  }

  private func handleGenerationEvent(_ event: GenerationEvent) {
    switch event {
    case .tokens(let text, let count):
      var message = messages.removeLast()
      let now = Date()
      if let last = message.lastSpeedMeasurementAt {
        let elapsed = now.timeIntervalSince(last)
        if elapsed > 0 { message.tokensPerSecond = Double(count) / elapsed }
      }
      message.lastSpeedMeasurementAt = now
      message.text += text
      message.tokenCount += count
      message.dateUpdated = now
      messages.append(message)

    case .thinkStart(let pendingText, let pendingCount):
      var message = messages.removeLast()
      let now = Date()
      if let last = message.lastSpeedMeasurementAt, now.timeIntervalSince(last) > 0 {
        message.tokensPerSecond = Double(pendingCount) / now.timeIntervalSince(last)
      }
      message.lastSpeedMeasurementAt = now
      message.text += pendingText
      message.text += message.text.isEmpty ? "Thinking...\n\n" : "\n\nThinking...\n\n"
      message.tokenCount += pendingCount + 1
      message.dateUpdated = now
      messages.append(message)

    case .thinkEnd(let pendingText, let pendingCount):
      var message = messages.removeLast()
      let now = Date()
      if let last = message.lastSpeedMeasurementAt, now.timeIntervalSince(last) > 0 {
        message.tokensPerSecond = Double(pendingCount) / now.timeIntervalSince(last)
      }
      message.lastSpeedMeasurementAt = now
      message.text += pendingText
      message.text += "\n\nFinished thinking.\n\n"
      message.tokenCount += pendingCount + 1
      message.dateUpdated = now
      messages.append(message)
    }
  }

  private func outputMessageType(for modelType: ModelType) -> MessageType {
    switch modelType {
    case .gemma3:   return .gemma3generated
    case .llama:    return .llamagenerated
    case .llava:    return .llavagenerated
    case .phi4:     return .phi4generated
    case .qwen3:    return .qwengenerated
    case .qwen3_5:  return .qwen3_5generated
    case .smollm3:  return .smollm3generated
    case .voxtral:  return .voxtralgenerated
    }
  }

  // MARK: - Voice Clone

  func toggleVoiceClone() {
    if voiceCloneEnabled {
      voiceCloneEnabled = false
      return
    }
    guard !isLoadingVoiceClone else { return }
    isLoadingVoiceClone = true
    Task {
      await speechManager.loadVoiceCloneModel(backend: voiceCloneBackendType)
      guard speechManager.isVoiceCloneLoaded else {
        print("[ChatViewModel] ❌ Voice clone model failed to load — aborting")
        isLoadingVoiceClone = false
        return
      }
      guard let url = Bundle.main.url(forResource: "voice_clone", withExtension: "m4a") else {
        print("[ChatViewModel] ❌ voice_clone.m4a not found in bundle — add it to the Xcode target")
        isLoadingVoiceClone = false
        return
      }
      await speechManager.setReferenceVoice(url: url, referenceText: voiceCloneReferenceText)
      guard speechManager.isEmbeddingReady else {
        print("[ChatViewModel] ❌ Speaker embedding extraction failed")
        isLoadingVoiceClone = false
        return
      }
      isLoadingVoiceClone = false
      voiceCloneEnabled = true
    }
  }

  // MARK: - Stop

  func stop() {
    shouldStopGenerating = true
  }

  // MARK: - Image

  func addSelectedImageMessage() {
    if let selectedImage {
      messages.append(Message(image: selectedImage))
    }
  }

  // MARK: - File Import

  func handleFileImportResult(_ pickerType: PickerType?, _ result: Result<[URL], Error>, onSuccess: (() -> Void)? = nil) {
    switch result {
    case .success(let urls):
      guard let url = urls.first, let pickerType else {
        withAnimation { messages.append(Message(type: .info, text: "Failed to select a file")) }
        return
      }
      llmService.runnerQueue.async { self.llmService.resetRunners() }
      switch pickerType {
      case .model:     resourceManager.modelPath = url.path
      case .tokenizer: resourceManager.tokenizerPath = url.path
      }
      lastPreloadedKey = nil
      if resourceManager.isModelValid && resourceManager.isTokenizerValid { onSuccess?() }
    case .failure(let error):
      withAnimation { messages.append(Message(type: .info, text: "Failed to select a file: \(error.localizedDescription)")) }
    }
  }

  // MARK: - Model Preloading

  func currentPreloadKey() -> String? {
    guard resourceManager.isModelValid && resourceManager.isTokenizerValid else { return nil }
    return resourceManager.modelPath + "|" + resourceManager.tokenizerPath
  }

  func loadModelIfNeededAsync() {
    let modelPath = resourceManager.modelPath
    let tokenizerPath = resourceManager.tokenizerPath
    llmService.runnerQueue.async {
      try? self.llmService.loadModelIfNeededSync(modelPath: modelPath, tokenizerPath: tokenizerPath)
    }
  }
}
