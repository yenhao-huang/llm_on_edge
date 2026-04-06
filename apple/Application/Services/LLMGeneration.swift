/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorchLLM
import Foundation
import SwiftUI

// MARK: - Runner Holder

class RunnerHolder {
  var textRunner: TextRunner?
  var multimodalRunner: MultimodalRunner?
}

// MARK: - Model Type

enum ModelType {
  case gemma3
  case llama
  case llava
  case qwen3
  case qwen3_5
  case phi4
  case smollm3
  case voxtral

  static func fromPath(_ path: String) -> ModelType {
    let filename = (path as NSString).lastPathComponent.lowercased()
    if filename.hasPrefix("gemma3") { return .gemma3 }
    else if filename.hasPrefix("llama") { return .llama }
    else if filename.hasPrefix("llava") { return .llava }
    else if filename.hasPrefix("qwen3.5") || filename.hasPrefix("qwen3_5") { return .qwen3_5 }
    else if filename.hasPrefix("qwen3") { return .qwen3 }
    else if filename.hasPrefix("phi4") { return .phi4 }
    else if filename.contains("smollm3") { return .smollm3 }
    else if filename.hasPrefix("voxtral") { return .voxtral }
    print("Unknown model type in path: \(path).")
    exit(1)
  }
}

// MARK: - Generation Events

enum GenerationEvent {
  case tokens(text: String, count: Int)
  case thinkStart(pendingText: String, pendingCount: Int)
  case thinkEnd(pendingText: String, pendingCount: Int)
}

// MARK: - Generation Errors

enum LLMGenerationError: LocalizedError {
  case audioNotFound
  case invalidAudioData

  var errorDescription: String? {
    switch self {
    case .audioNotFound: return "Audio not found"
    case .invalidAudioData: return "Invalid audio data"
    }
  }
}

// MARK: - Output Parser

enum OutputParser {
  /// Characters to strip from generated text before display.
  static let blacklist: Set<Character> = ["*", "_", "`", "~"]

  static func parse(_ text: String) -> String {
    String(text.filter { !blacklist.contains($0) })
  }
}

// MARK: - LLM Generation Service

class LLMGenerationService {
  let runnerHolder = RunnerHolder()
  let runnerQueue = DispatchQueue(label: "org.pytorch.executorch.etllm")
  private var shouldStopShowingToken = false

  func resetRunners() {
    runnerHolder.textRunner = nil
    runnerHolder.multimodalRunner = nil
  }

  /// Load model synchronously. Must be called from `runnerQueue`.
  /// Returns load duration, or `nil` if model was already loaded.
  func loadModelIfNeededSync(modelPath: String, tokenizerPath: String) throws -> TimeInterval? {
    let modelType = ModelType.fromPath(modelPath)
    switch modelType {
    case .llama:
      runnerHolder.textRunner = runnerHolder.textRunner ?? TextRunner(
        modelPath: modelPath,
        tokenizerPath: tokenizerPath,
        specialTokens: [
          "<|begin_of_text|>", "<|end_of_text|>",
          "<|reserved_special_token_0|>", "<|reserved_special_token_1|>",
          "<|finetune_right_pad_id|>", "<|step_id|>",
          "<|start_header_id|>", "<|end_header_id|>",
          "<|eom_id|>", "<|eot_id|>", "<|python_tag|>"
        ] + (2..<256).map { "<|reserved_special_token_\($0)|>" }
      )
    case .qwen3, .qwen3_5, .phi4, .smollm3:
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

    if let runner = runnerHolder.textRunner, !runner.isLoaded() {
      let start = Date()
      try runner.load()
      return Date().timeIntervalSince(start)
    } else if let runner = runnerHolder.multimodalRunner, !runner.isLoaded() {
      let start = Date()
      try runner.load()
      return Date().timeIntervalSince(start)
    }
    return nil
  }

  /// Run generation synchronously. Must be called from `runnerQueue`.
  /// Streams events via `onEvent` (called on runnerQueue — caller must dispatch to main if needed).
  func generate(
    text: String,
    modelPath: String,
    thinkingMode: Bool,
    image: PlatformImage?,
    onEvent: @escaping (GenerationEvent) -> Void,
    shouldStop: @escaping () -> Bool
  ) throws {
    shouldStopShowingToken = false
    let modelType = ModelType.fromPath(modelPath)
    let sequenceLength = (modelType == .llava || modelType == .gemma3 || modelType == .voxtral)
      ? 768
      : ((modelType == .llama || modelType == .phi4) ? 128 : 768)
    var tokens: [String] = []

    func flush() -> (text: String, count: Int) {
      let t = OutputParser.parse(tokens.joined()); let c = tokens.count; tokens = []; return (t, c)
    }

    if (modelType == .llava || modelType == .gemma3), let image = image {
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
      } else {
        let sideSize: CGFloat = 896
        inputs = [
          MultimodalInput(systemPrompt + "<start_of_image>"),
          MultimodalInput(image.asNormalizedImage(sideSize)),
          MultimodalInput("<end_of_image>" + String(format: Constants.gemma3PromptTemplate, text))
        ]
        config.sequenceLength = 768
        promptToIgnore = String(format: Constants.gemma3PromptTemplate, text)
      }
      try runnerHolder.multimodalRunner?.generate(inputs, config) { [self] token in
        if token == promptToIgnore { return }
        if (modelType == .gemma3 && token == "<end_of_turn>") ||
           (modelType == .llava && token == "</s>") {
          runnerHolder.multimodalRunner?.stop(); return
        }
        tokens.append(token)
        if tokens.count > 2 { let (t, c) = flush(); onEvent(.tokens(text: t, count: c)) }
        if shouldStop() { runnerHolder.multimodalRunner?.stop() }
      }

    } else if modelType == .voxtral {
      guard let audioPath = Bundle.main.path(forResource: "voxtral_input_features", ofType: "bin") else {
        throw LLMGenerationError.audioNotFound
      }
      let audioData = try Data(contentsOf: URL(fileURLWithPath: audioPath), options: .mappedIfSafe)
      let floatByteCount = MemoryLayout<Float>.size
      guard audioData.count % floatByteCount == 0 else { throw LLMGenerationError.invalidAudioData }
      let bins = 128, frames = 3000
      let batchSize = audioData.count / floatByteCount / (bins * frames)
      try runnerHolder.multimodalRunner?.generate([
        MultimodalInput("<s>[INST][BEGIN_AUDIO]"),
        MultimodalInput(Audio(float: audioData, batchSize: batchSize, bins: bins, frames: frames)),
        MultimodalInput(String(format: Constants.voxtralPromptTemplate, text))
      ], Config { $0.maximumNewTokens = 256 }) { [self] token in
        tokens.append(token)
        if tokens.count > 2 { let (t, c) = flush(); onEvent(.tokens(text: t, count: c)) }
        if shouldStop() { runnerHolder.multimodalRunner?.stop() }
      }

    } else if modelType == .llava || modelType == .gemma3 {
      // Multimodal without image
      let systemPrompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
      var inputs: [MultimodalInput]
      var config = Config()
      var promptToIgnore: String
      if modelType == .llava {
        inputs = [MultimodalInput(systemPrompt), MultimodalInput(String(format: Constants.llavaPromptTemplate, text))]
        config.sequenceLength = 768
        promptToIgnore = String(format: Constants.llavaPromptTemplate, text)
      } else {
        inputs = [MultimodalInput(systemPrompt), MultimodalInput(String(format: Constants.gemma3PromptTemplate, text))]
        config.sequenceLength = 768
        promptToIgnore = String(format: Constants.gemma3PromptTemplate, text)
      }
      try runnerHolder.multimodalRunner?.generate(inputs, config) { [self] token in
        if token == promptToIgnore { return }
        if (modelType == .gemma3 && token == "<end_of_turn>") ||
           (modelType == .llava && token == "</s>") {
          runnerHolder.multimodalRunner?.stop(); return
        }
        tokens.append(token)
        if tokens.count > 2 { let (t, c) = flush(); onEvent(.tokens(text: t, count: c)) }
        if shouldStop() { runnerHolder.multimodalRunner?.stop() }
      }

    } else {
      // Text-only models
      let formattedPrompt: String
      switch modelType {
      case .gemma3:  formattedPrompt = String(format: Constants.gemma3PromptTemplate, text)
      case .llama:   formattedPrompt = String(format: Constants.llama3PromptTemplate, text)
      case .llava:   formattedPrompt = String(format: Constants.llavaPromptTemplate, text)
      case .phi4:    formattedPrompt = String(format: Constants.phi4PromptTemplate, text)
      case .qwen3, .qwen3_5:
        let base = String(format: Constants.qwen3PromptTemplate, text)
        formattedPrompt = thinkingMode
          ? base.replacingOccurrences(of: "<think>\n\n</think>\n\n\n", with: "")
          : base
      case .smollm3: formattedPrompt = String(format: Constants.smolLm3PromptTemplate, text)
      case .voxtral: formattedPrompt = String(format: Constants.voxtralPromptTemplate, text)
      }

      try runnerHolder.textRunner?.generate(formattedPrompt, Config { $0.sequenceLength = sequenceLength }) { [self] token in
        if modelType == .gemma3 && token == "<end_of_turn>" {
          runnerHolder.textRunner?.stop(); return
        }
        if modelType == .llama && (token == "<|eot_id|>" || token == "<|end_of_text|>") {
          runnerHolder.textRunner?.stop(); return
        }
        if modelType == .phi4 && token == "<|end|>" {
          runnerHolder.textRunner?.stop(); return
        }
        if (modelType == .qwen3 || modelType == .qwen3_5 || modelType == .smollm3) && token == "<|im_end|>" {
          runnerHolder.textRunner?.stop(); return
        }
        if token == formattedPrompt { return }
        if token == "<|eot_id|>" { shouldStopShowingToken = true; return }
        if token == "<|im_end|>" { return }

        if token == "<think>" {
          let (t, c) = flush()
          onEvent(.thinkStart(pendingText: t, pendingCount: c))
          return
        }
        if token == "</think>" {
          let (t, c) = flush()
          onEvent(.thinkEnd(pendingText: t, pendingCount: c))
          return
        }

        guard !shouldStopShowingToken else { return }
        tokens.append(token.trimmingCharacters(in: .newlines))
        if tokens.count > 2 { let (t, c) = flush(); onEvent(.tokens(text: t, count: c)) }
        if shouldStop() { runnerHolder.textRunner?.stop() }
      }
    }
  }
}
