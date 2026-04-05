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

// MARK: - Supporting Types

class RunnerHolder {
  var textRunner: TextRunner?
  var multimodalRunner: MultimodalRunner?
}

enum ModelType {
  case gemma3
  case llama
  case llava
  case qwen3
  case phi4
  case smollm3
  case voxtral

  static func fromPath(_ path: String) -> ModelType {
    let filename = (path as NSString).lastPathComponent.lowercased()
    if filename.hasPrefix("gemma3") {
      return .gemma3
    } else if filename.hasPrefix("llama") {
      return .llama
    } else if filename.hasPrefix("llava") {
      return .llava
    } else if filename.hasPrefix("qwen3") {
      return .qwen3
    } else if filename.hasPrefix("phi4") {
      return .phi4
    } else if filename.contains("smollm3") {
      return .smollm3
    } else if filename.hasPrefix("voxtral") {
      return .voxtral
    }
    print("Unknown model type in path: \(path).")
    exit(1)
  }
}

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
  @Published var selectedImage: PlatformImage?

  let resourceManager = ResourceManager()
  let resourceMonitor = ResourceMonitor()
  let logManager = LogManager()
  let speechManager: SpeechManager
  let speechRecognitionManager: SpeechRecognitionManager

  private var shouldStopShowingToken = false
  private let runnerHolder = RunnerHolder()
  private let runnerQueue = DispatchQueue(label: "org.pytorch.executorch.etllm")
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

    // Live conversation restart logic — platform-independent, lives in ViewModel
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

  // MARK: - Actions

  func generate() {
    guard !prompt.isEmpty else { return }
    guard speechRecognitionManager.recordingState != .recording else { return }
    isGenerating = true
    shouldStopGenerating = false
    shouldStopShowingToken = false
    let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    let modelPath = resourceManager.modelPath
    let tokenizerPath = resourceManager.tokenizerPath
    let modelType = ModelType.fromPath(modelPath)
    let sequenceLength = (modelType == .llava || modelType == .gemma3 || modelType == .voxtral) ? 768 : ((modelType == .llama || modelType == .phi4) ? 128 : 768)
    prompt = ""
    messages.append(Message(text: text))
    let outputType: MessageType = {
      switch modelType {
      case .gemma3: return .gemma3generated
      case .llama: return .llamagenerated
      case .llava: return .llavagenerated
      case .phi4: return .phi4generated
      case .qwen3: return .qwengenerated
      case .smollm3: return .smollm3generated
      case .voxtral: return .voxtralgenerated
      }
    }()
    messages.append(Message(type: outputType))

    runnerQueue.async { [weak self] in
      guard let self else { return }

      defer {
        DispatchQueue.main.async {
          self.isGenerating = false
          self.selectedImage = nil
          if self.autoSpeak, let last = self.messages.last {
            switch last.type {
            case .llamagenerated, .llavagenerated, .qwengenerated, .phi4generated, .gemma3generated, .smollm3generated, .voxtralgenerated:
              if !last.text.isEmpty {
                self.speechManager.speak(last.text)
              }
            default:
              break
            }
          }
        }
      }

      self.loadModelIfNeededSync(reportToUI: true)

      guard !self.shouldStopGenerating else {
        DispatchQueue.main.async {
          withAnimation {
            _ = self.messages.removeLast()
          }
        }
        return
      }
      defer {
        self.runnerHolder.textRunner?.reset()
        self.runnerHolder.multimodalRunner?.reset()
      }

      do {
        var tokens: [String] = []

        if (modelType == .llava || modelType == .gemma3), let image = self.selectedImage {
          let systemPrompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
          var inputs: [MultimodalInput] = []
          var config = Config()
          var promptToIgnore = ""
          if modelType == .llava {
            let sideSize: CGFloat = 336
            inputs = [
              MultimodalInput(systemPrompt),
              MultimodalInput(image.asImage(sideSize)),
              MultimodalInput(String(format: Constants.llavaPromptTemplate, text))
            ]
            config.sequenceLength = 768
            promptToIgnore = String(format: Constants.llavaPromptTemplate, text)
          }
          if modelType == .gemma3 {
            let sideSize: CGFloat = 896
            inputs = [
              MultimodalInput(systemPrompt + "<start_of_image>"),
              MultimodalInput(image.asNormalizedImage(sideSize)),
              MultimodalInput("<end_of_image>" + String(format: Constants.gemma3PromptTemplate, text))
            ]
            config.sequenceLength = 768
            promptToIgnore = String(format: Constants.gemma3PromptTemplate, text)
          }

          try self.runnerHolder.multimodalRunner?.generate(inputs, config) { token in
            if token == promptToIgnore { return }
            if modelType == .gemma3 && token == "<end_of_turn>" {
              self.shouldStopGenerating = true
              self.runnerHolder.multimodalRunner?.stop()
              return
            }
            if modelType == .llava && token == "</s>" {
              self.shouldStopGenerating = true
              self.runnerHolder.multimodalRunner?.stop()
              return
            }
            tokens.append(token)
            if tokens.count > 2 {
              let streamedText = tokens.joined()
              let streamedCount = tokens.count
              tokens = []
              DispatchQueue.main.async {
                var message = self.messages.removeLast()
                let now = Date()
                if let last = message.lastSpeedMeasurementAt {
                  let elapsedSeconds = now.timeIntervalSince(last)
                  if elapsedSeconds > 0 {
                    message.tokensPerSecond = Double(streamedCount) / elapsedSeconds
                  }
                }
                message.lastSpeedMeasurementAt = now
                message.text += streamedText
                message.tokenCount += streamedCount
                message.dateUpdated = now
                self.messages.append(message)
              }
            }
            if self.shouldStopGenerating {
              self.runnerHolder.multimodalRunner?.stop()
            }
          }
        } else if modelType == .voxtral {
          guard let audioPath = Bundle.main.path(forResource: "voxtral_input_features", ofType: "bin") else {
            DispatchQueue.main.async {
              withAnimation {
                var message = self.messages.removeLast()
                message.type = .info
                message.text = "Audio not found"
                self.messages.append(message)
              }
            }
            return
          }
          var audioData = try Data(contentsOf: URL(fileURLWithPath: audioPath), options: .mappedIfSafe)
          let floatByteCount = MemoryLayout<Float>.size
          guard audioData.count % floatByteCount == 0 else {
            DispatchQueue.main.async {
              withAnimation {
                var message = self.messages.removeLast()
                message.type = .info
                message.text = "Invalid audio data"
                self.messages.append(message)
              }
            }
            return
          }
          let bins = 128
          let frames = 3000
          let batchSize = audioData.count / floatByteCount / (bins * frames)

          try self.runnerHolder.multimodalRunner?.generate([
            MultimodalInput("<s>[INST][BEGIN_AUDIO]"),
            MultimodalInput(Audio(float: audioData, batchSize: batchSize, bins: bins, frames: frames)),
            MultimodalInput(String(format: Constants.voxtralPromptTemplate, text))
          ], Config {
            $0.maximumNewTokens = 256
          }) { token in
            tokens.append(token)
            if tokens.count > 2 {
              let streamedText = tokens.joined()
              let streamedCount = tokens.count
              tokens = []
              DispatchQueue.main.async {
                var message = self.messages.removeLast()
                let now = Date()
                if let last = message.lastSpeedMeasurementAt {
                  let elapsedSeconds = now.timeIntervalSince(last)
                  if elapsedSeconds > 0 {
                    message.tokensPerSecond = Double(streamedCount) / elapsedSeconds
                  }
                }
                message.lastSpeedMeasurementAt = now
                message.text += streamedText
                message.tokenCount += streamedCount
                message.dateUpdated = now
                self.messages.append(message)
              }
            }
            if self.shouldStopGenerating {
              self.runnerHolder.multimodalRunner?.stop()
            }
          }
        } else if (modelType == .llava || modelType == .gemma3) {
          let systemPrompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
          var inputs: [MultimodalInput] = []
          var config = Config()
          var promptToIgnore = ""
          if modelType == .llava {
            inputs = [
              MultimodalInput(systemPrompt),
              MultimodalInput(String(format: Constants.llavaPromptTemplate, text))
            ]
            config.sequenceLength = 768
            promptToIgnore = String(format: Constants.llavaPromptTemplate, text)
          } else {
            inputs = [
              MultimodalInput(systemPrompt),
              MultimodalInput(String(format: Constants.gemma3PromptTemplate, text))
            ]
            config.sequenceLength = 768
            promptToIgnore = String(format: Constants.gemma3PromptTemplate, text)
          }
          try self.runnerHolder.multimodalRunner?.generate(inputs, config) { token in
            if token == promptToIgnore { return }
            if modelType == .gemma3 && token == "<end_of_turn>" {
              self.shouldStopGenerating = true
              self.runnerHolder.multimodalRunner?.stop()
              return
            }
            if modelType == .llava && token == "</s>" {
              self.shouldStopGenerating = true
              self.runnerHolder.multimodalRunner?.stop()
              return
            }
            tokens.append(token)
            if tokens.count > 2 {
              let streamedText = tokens.joined()
              let streamedCount = tokens.count
              tokens = []
              DispatchQueue.main.async {
                var message = self.messages.removeLast()
                let now = Date()
                if let last = message.lastSpeedMeasurementAt {
                  let elapsedSeconds = now.timeIntervalSince(last)
                  if elapsedSeconds > 0 {
                    message.tokensPerSecond = Double(streamedCount) / elapsedSeconds
                  }
                }
                message.lastSpeedMeasurementAt = now
                message.text += streamedText
                message.tokenCount += streamedCount
                message.dateUpdated = now
                self.messages.append(message)
              }
            }
            if self.shouldStopGenerating {
              self.runnerHolder.multimodalRunner?.stop()
            }
          }
        } else {
          let formattedPrompt: String
          switch modelType {
          case .gemma3:
            formattedPrompt = String(format: Constants.gemma3PromptTemplate, text)
          case .llama:
            formattedPrompt = String(format: Constants.llama3PromptTemplate, text)
          case .llava:
            formattedPrompt = String(format: Constants.llavaPromptTemplate, text)
          case .phi4:
            formattedPrompt = String(format: Constants.phi4PromptTemplate, text)
          case .qwen3:
            let basePrompt = String(format: Constants.qwen3PromptTemplate, text)
            formattedPrompt = self.thinkingMode ? basePrompt.replacingOccurrences(of: "<think>\n\n</think>\n\n\n", with: "") : basePrompt
          case .smollm3:
            formattedPrompt = String(format: Constants.smolLm3PromptTemplate, text)
          case .voxtral:
            formattedPrompt = String(format: Constants.voxtralPromptTemplate, text)
          }
          try self.runnerHolder.textRunner?.generate(formattedPrompt, Config {
            $0.sequenceLength = sequenceLength
          }) { token in
            if modelType == .gemma3 && token == "<end_of_turn>" {
              self.shouldStopGenerating = true
              self.runnerHolder.textRunner?.stop()
            }
            if modelType == .llama && (token == "<|eot_id|>" || token == "<|end_of_text|>") {
              self.shouldStopGenerating = true
              self.runnerHolder.textRunner?.stop()
              return
            }
            if modelType == .phi4 && token == "<|end|>" {
              self.shouldStopGenerating = true
              self.runnerHolder.textRunner?.stop()
              return
            }
            if (modelType == .qwen3 || modelType == .smollm3) && token == "<|im_end|>" {
              self.shouldStopGenerating = true
              self.runnerHolder.textRunner?.stop()
              return
            }
            if token != formattedPrompt {
              if token == "<|eot_id|>" {
                self.shouldStopShowingToken = true
              } else if token == "<|im_end|>" {
              } else if token == "<think>" {
                let flushedText = tokens.joined()
                let flushedCount = tokens.count
                tokens = []
                DispatchQueue.main.async {
                  var message = self.messages.removeLast()
                  let now = Date()
                  if let last = message.lastSpeedMeasurementAt {
                    let elapsedSeconds = now.timeIntervalSince(last)
                    if elapsedSeconds > 0 {
                      message.tokensPerSecond = Double(flushedCount) / elapsedSeconds
                    }
                  }
                  message.lastSpeedMeasurementAt = now
                  message.text += flushedText
                  message.text += message.text.isEmpty ? "Thinking...\n\n" : "\n\nThinking...\n\n"
                  message.tokenCount += flushedCount + 1
                  message.dateUpdated = now
                  self.messages.append(message)
                }
              } else if token == "</think>" {
                let flushedText = tokens.joined()
                let flushedCount = tokens.count
                tokens = []
                DispatchQueue.main.async {
                  var message = self.messages.removeLast()
                  let now = Date()
                  if let last = message.lastSpeedMeasurementAt {
                    let elapsedSeconds = now.timeIntervalSince(last)
                    if elapsedSeconds > 0 {
                      message.tokensPerSecond = Double(flushedCount) / elapsedSeconds
                    }
                  }
                  message.lastSpeedMeasurementAt = now
                  message.text += flushedText
                  message.text += "\n\nFinished thinking.\n\n"
                  message.tokenCount += flushedCount + 1
                  message.dateUpdated = now
                  self.messages.append(message)
                }
              } else {
                tokens.append(token.trimmingCharacters(in: .newlines))
                if tokens.count > 2 {
                  let streamedText = tokens.joined()
                  let streamedCount = tokens.count
                  tokens = []
                  DispatchQueue.main.async {
                    var message = self.messages.removeLast()
                    let now = Date()
                    if let last = message.lastSpeedMeasurementAt {
                      let elapsedSeconds = now.timeIntervalSince(last)
                      if elapsedSeconds > 0 {
                        message.tokensPerSecond = Double(streamedCount) / elapsedSeconds
                      }
                    }
                    message.lastSpeedMeasurementAt = now
                    message.text += streamedText
                    message.tokenCount += streamedCount
                    message.dateUpdated = now
                    self.messages.append(message)
                  }
                }
                if self.shouldStopGenerating {
                  self.runnerHolder.textRunner?.stop()
                }
              }
            }
          }
        }
      } catch {
        DispatchQueue.main.async {
          withAnimation {
            var message = self.messages.removeLast()
            message.type = .info
            message.text = "Text generation failed: error \((error as NSError).code)"
            self.messages.append(message)
          }
        }
      }
    }
  }

  func stop() {
    shouldStopGenerating = true
  }

  func addSelectedImageMessage() {
    if let selectedImage {
      messages.append(Message(image: selectedImage))
    }
  }

  func handleFileImportResult(_ pickerType: PickerType?, _ result: Result<[URL], Error>, onSuccess: (() -> Void)? = nil) {
    switch result {
    case .success(let urls):
      guard let url = urls.first, let pickerType else {
        withAnimation {
          messages.append(Message(type: .info, text: "Failed to select a file"))
        }
        return
      }
      runnerQueue.async {
        self.runnerHolder.textRunner = nil
        self.runnerHolder.multimodalRunner = nil
      }
      switch pickerType {
      case .model:
        resourceManager.modelPath = url.path
      case .tokenizer:
        resourceManager.tokenizerPath = url.path
      }
      lastPreloadedKey = nil
      if resourceManager.isModelValid && resourceManager.isTokenizerValid {
        onSuccess?()
      }
    case .failure(let error):
      withAnimation {
        messages.append(Message(type: .info, text: "Failed to select a file: \(error.localizedDescription)"))
      }
    }
  }

  // MARK: - Model Loading

  func currentPreloadKey() -> String? {
    guard resourceManager.isModelValid && resourceManager.isTokenizerValid else { return nil }
    return resourceManager.modelPath + "|" + resourceManager.tokenizerPath
  }

  func loadModelIfNeededAsync(reportToUI: Bool) {
    runnerQueue.async {
      self.loadModelIfNeededSync(reportToUI: reportToUI)
    }
  }

  private func loadModelIfNeededSync(reportToUI: Bool) {
    guard resourceManager.isModelValid && resourceManager.isTokenizerValid else { return }
    let modelPath = resourceManager.modelPath
    let tokenizerPath = resourceManager.tokenizerPath
    let modelType = ModelType.fromPath(modelPath)

    switch modelType {
    case .llama:
      runnerHolder.textRunner = runnerHolder.textRunner ?? TextRunner(
        modelPath: modelPath,
        tokenizerPath: tokenizerPath,
        specialTokens: [
          "<|begin_of_text|>",
          "<|end_of_text|>",
          "<|reserved_special_token_0|>",
          "<|reserved_special_token_1|>",
          "<|finetune_right_pad_id|>",
          "<|step_id|>",
          "<|start_header_id|>",
          "<|end_header_id|>",
          "<|eom_id|>",
          "<|eot_id|>",
          "<|python_tag|>"
        ] + (2..<256).map { "<|reserved_special_token_\($0)|>" }
      )
    case .qwen3, .phi4, .smollm3:
      runnerHolder.textRunner = runnerHolder.textRunner ?? TextRunner(
        modelPath: modelPath,
        tokenizerPath: tokenizerPath
      )
    case .llava, .gemma3, .voxtral:
      runnerHolder.multimodalRunner = runnerHolder.multimodalRunner ?? MultimodalRunner(
        modelPath: modelPath,
        tokenizerPath: tokenizerPath
      )
    }

    if (modelType == .llama || modelType == .qwen3 || modelType == .phi4 || modelType == .smollm3),
       let runner = runnerHolder.textRunner, !runner.isLoaded() {
      var err: Error?
      let start = Date()
      do { try runner.load() } catch { err = error }
      let dur = Date().timeIntervalSince(start)
      if reportToUI {
        DispatchQueue.main.async {
          withAnimation {
            var message = self.messages.removeLast()
            message.type = .info
            if let err {
              message.text = "Model loading failed: error \((err as NSError).code)"
            } else {
              message.text = "Model loaded in \(String(format: "%.2f", dur)) s"
            }
            self.messages.append(message)
            if err == nil {
              let outputType: MessageType = {
                switch modelType {
                case .gemma3: return .gemma3generated
                case .llama: return .llamagenerated
                case .llava: return .llavagenerated
                case .phi4: return .phi4generated
                case .qwen3: return .qwengenerated
                case .smollm3: return .smollm3generated
                case .voxtral: return .voxtralgenerated
                }
              }()
              self.messages.append(Message(type: outputType))
            }
          }
        }
      }
    } else if let runner = runnerHolder.multimodalRunner, !runner.isLoaded() {
      var err: Error?
      let start = Date()
      do { try runner.load() } catch { err = error }
      let dur = Date().timeIntervalSince(start)
      if reportToUI {
        DispatchQueue.main.async {
          withAnimation {
            var message = self.messages.removeLast()
            message.type = .info
            if let err {
              message.text = "Model loading failed: error \((err as NSError).code)"
            } else {
              message.text = "Model loaded in \(String(format: "%.2f", dur)) s"
            }
            self.messages.append(message)
            if err == nil {
              let outputType: MessageType = {
                switch modelType {
                case .gemma3: return .gemma3generated
                case .llama: return .llamagenerated
                case .llava: return .llavagenerated
                case .phi4: return .phi4generated
                case .qwen3: return .qwengenerated
                case .smollm3: return .smollm3generated
                case .voxtral: return .voxtralgenerated
                }
              }()
              self.messages.append(Message(type: outputType))
            }
          }
        }
      }
    }
  }
}
