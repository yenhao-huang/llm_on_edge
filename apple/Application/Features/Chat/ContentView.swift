/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI
import UniformTypeIdentifiers

#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

struct ContentView: View {
  @StateObject private var viewModel = ChatViewModel()

  // UI-only state (not business logic)
  @State private var showingLogs = false
  @State private var pickerType: PickerType?
  @State private var showThinkingModeNotification = false
  @State private var showingSettings = false
  @FocusState private var textFieldFocused: Bool

  #if os(iOS)
  @State private var isImagePickerPresented = false
  @State private var imagePickerSourceType: UIImagePickerController.SourceType = .photoLibrary
  #endif

  var body: some View {
    #if os(macOS)
    macOSBody
    #else
    iOSBody
    #endif
  }

  // MARK: - macOS

  #if os(macOS)
  @ViewBuilder
  private var macOSBody: some View {
    NavigationSplitView {
      VStack(alignment: .leading, spacing: 12) {
        Text("Model")
          .font(.headline)
          .foregroundColor(.secondary)
          .padding(.top, 8)

        Button(action: { pickerType = .model }) {
          HStack {
            Image(systemName: "cpu")
            Text(viewModel.resourceManager.isModelValid ? viewModel.resourceManager.modelName : "Select Model...")
              .lineLimit(1)
              .truncationMode(.middle)
            Spacer()
          }
          .padding(8)
          .background(Color.gray.opacity(0.1))
          .cornerRadius(8)
        }
        .buttonStyle(.plain)

        Button(action: { pickerType = .tokenizer }) {
          HStack {
            Image(systemName: "doc.text")
            Text(viewModel.resourceManager.isTokenizerValid ? viewModel.resourceManager.tokenizerName : "Select Tokenizer...")
              .lineLimit(1)
              .truncationMode(.middle)
            Spacer()
          }
          .padding(8)
          .background(Color.gray.opacity(0.1))
          .cornerRadius(8)
        }
        .buttonStyle(.plain)

        Divider().padding(.vertical, 8)

        Text("Memory")
          .font(.headline)
          .foregroundColor(.secondary)

        VStack(alignment: .leading, spacing: 4) {
          HStack {
            Text("Used:")
            Spacer()
            Text("\(viewModel.resourceMonitor.usedMemory) MB").monospacedDigit()
          }
          HStack {
            Text("Available:")
            Spacer()
            Text("\(viewModel.resourceMonitor.availableMemory) MB").monospacedDigit()
          }
        }
        .font(.caption)
        .onAppear { viewModel.resourceMonitor.start() }
        .onDisappear { viewModel.resourceMonitor.stop() }

        Divider().padding(.vertical, 8)

        Button(action: { showingLogs = true }) {
          HStack {
            Image(systemName: "list.bullet.rectangle")
            Text("Logs")
            Spacer()
          }
          .padding(8)
          .background(Color.gray.opacity(0.1))
          .cornerRadius(8)
        }
        .buttonStyle(.plain)

        Button(action: { viewModel.autoSpeak.toggle() }) {
          HStack {
            Image(systemName: viewModel.autoSpeak ? "speaker.wave.2.fill" : "speaker.slash")
              .foregroundColor(viewModel.autoSpeak ? .blue : .gray)
            Text("Auto Speak")
            Spacer()
          }
          .padding(8)
          .background(Color.gray.opacity(0.1))
          .cornerRadius(8)
        }
        .buttonStyle(.plain)

        Button(action: { viewModel.toggleVoiceClone() }) {
          HStack {
            VoiceCloneButtonView(
              isLoading: viewModel.isLoadingVoiceClone,
              isEnabled: viewModel.voiceCloneEnabled
            )
            .frame(width: 24, height: 24)
            Text("Voice Clone")
            Spacer()
          }
          .padding(8)
          .background(viewModel.voiceCloneEnabled ? Color.purple.opacity(0.1) : Color.gray.opacity(0.1))
          .cornerRadius(8)
        }
        .buttonStyle(.plain)

        Button(action: {
          if viewModel.speechRecognitionManager.isLiveMode {
            viewModel.speechRecognitionManager.stopLiveMode()
          } else {
            if viewModel.speechManager.isSpeaking { viewModel.speechManager.stop() }
            viewModel.speechRecognitionManager.startLiveMode()
          }
        }) {
          HStack {
            Image(systemName: viewModel.speechRecognitionManager.isLiveMode ? "waveform.circle.fill" : "waveform.circle")
              .foregroundColor(viewModel.speechRecognitionManager.isLiveMode ? .green : .gray)
            Text("Live Conversation")
            Spacer()
            if viewModel.speechRecognitionManager.isLiveMode {
              Circle()
                .fill(viewModel.speechRecognitionManager.recordingState == .recording ? Color.red : Color.orange)
                .frame(width: 8, height: 8)
            }
          }
          .padding(8)
          .background(viewModel.speechRecognitionManager.isLiveMode ? Color.green.opacity(0.1) : Color.gray.opacity(0.1))
          .cornerRadius(8)
        }
        .buttonStyle(.plain)

        Spacer()
      }
      .padding(.horizontal)
      .frame(minWidth: 200, maxWidth: 250)
      .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 280)
    } detail: {
      VStack(spacing: 0) {
        MessageListView(messages: $viewModel.messages, speechManager: viewModel.speechManager)
          .frame(maxWidth: .infinity, maxHeight: .infinity)

        HStack {
          Button(action: { selectImageOnMac() }) {
            Image(systemName: "photo.on.rectangle")
              .resizable()
              .scaledToFit()
              .frame(width: 24, height: 24)
          }
          .buttonStyle(.plain)

          Button(action: {
            if viewModel.speechManager.isSpeaking { viewModel.speechManager.stop() }
            viewModel.speechRecognitionManager.toggleRecording()
          }) {
            MicButtonView(state: viewModel.speechRecognitionManager.recordingState)
          }
          .buttonStyle(.plain)

          if viewModel.resourceManager.isModelValid && ModelType.fromPath(viewModel.resourceManager.modelPath) == .qwen3 {
            Button(action: {
              viewModel.thinkingMode.toggle()
              showThinkingModeNotification = true
              DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                showThinkingModeNotification = false
              }
            }) {
              Image(systemName: "brain")
                .resizable()
                .scaledToFit()
                .frame(width: 24, height: 24)
                .foregroundColor(viewModel.thinkingMode ? .blue : .gray)
            }
            .buttonStyle(.plain)
          }

          TextField(
            viewModel.resourceManager.isModelValid ? viewModel.resourceManager.isTokenizerValid ? "Prompt..." : "Select Tokenizer..." : "Select Model...",
            text: $viewModel.prompt,
            axis: .vertical
          )
          .padding(8)
          .background(Color.gray.opacity(0.1))
          .cornerRadius(20)
          .lineLimit(1...10)
          .overlay(
            RoundedRectangle(cornerRadius: 20)
              .stroke(viewModel.isInputEnabled ? Color.blue : Color.gray, lineWidth: 1)
          )
          .disabled(!viewModel.isInputEnabled)
          .focused($textFieldFocused)
          .onSubmit {
            if !viewModel.prompt.isEmpty && viewModel.isInputEnabled && !viewModel.isGenerating {
              viewModel.generate()
            }
          }

          Button(action: viewModel.isGenerating ? viewModel.stop : viewModel.generate) {
            Image(systemName: viewModel.isGenerating ? "stop.circle" : "arrowshape.up.circle.fill")
              .resizable()
              .aspectRatio(contentMode: .fit)
              .frame(height: 28)
          }
          .buttonStyle(.plain)
          .disabled(viewModel.isGenerating ? viewModel.shouldStopGenerating : (!viewModel.isInputEnabled || viewModel.prompt.isEmpty))
        }
        .padding(10)
      }
      .overlay {
        if showThinkingModeNotification {
          Text(viewModel.thinkingMode ? "Thinking mode enabled" : "Thinking mode disabled")
            .padding(8)
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(8)
            .transition(.opacity)
            .animation(.easeInOut(duration: 0.2), value: showThinkingModeNotification)
        }
      }
    }
    .sheet(isPresented: $showingLogs) {
      VStack(spacing: 0) {
        HStack {
          Text("Logs").font(.headline)
          Spacer()
          Button(action: { viewModel.logManager.clear() }) {
            Image(systemName: "trash")
          }
          .help("Clear logs")
          Button("Done") { showingLogs = false }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))

        Divider()
        LogView(logManager: viewModel.logManager)
      }
      .frame(minWidth: 600, minHeight: 400)
    }
    .fileImporter(
      isPresented: Binding<Bool>(
        get: { pickerType != nil },
        set: { if !$0 { pickerType = nil } }
      ),
      allowedContentTypes: allowedContentTypes(),
      allowsMultipleSelection: false
    ) { [pickerType] result in
      viewModel.handleFileImportResult(pickerType, result) {
        textFieldFocused = true
      }
    }
    .onAppear {
      viewModel.setup()
      viewModel.speechRecognitionManager.onTranscription = { transcribed in
        viewModel.prompt = transcribed
        textFieldFocused = false
        viewModel.generate()
      }
    }
  }

  private func selectImageOnMac() {
    let panel = NSOpenPanel()
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = false
    panel.canChooseFiles = true
    panel.allowedContentTypes = [.image, .png, .jpeg, .gif, .heic]
    panel.message = "Select an image"
    panel.prompt = "Select"
    if panel.runModal() == .OK {
      if let url = panel.url, let image = NSImage(contentsOf: url) {
        viewModel.selectedImage = image
        viewModel.addSelectedImageMessage()
      }
    }
  }
  #endif

  // MARK: - iOS

  #if os(iOS)
  @ViewBuilder
  private var iOSBody: some View {
    NavigationView {
      ZStack {
        VStack {
          if showingSettings {
            VStack(spacing: 20) {
              HStack {
                VStack(spacing: 10) {
                  Button(action: { pickerType = .model }) {
                    Label(
                      viewModel.resourceManager.isModelValid ? viewModel.resourceManager.modelName : "Select Model...",
                      systemImage: "doc"
                    )
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 300, alignment: .leading)
                  }
                  Button(action: { pickerType = .tokenizer }) {
                    Label(
                      viewModel.resourceManager.isTokenizerValid ? viewModel.resourceManager.tokenizerName : "Select Tokenizer...",
                      systemImage: "doc"
                    )
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 300, alignment: .leading)
                  }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)
                .fixedSize(horizontal: true, vertical: false)
                Spacer()
              }
              .padding()
            }
          }

          MessageListView(messages: $viewModel.messages, speechManager: viewModel.speechManager)
            .simultaneousGesture(
              DragGesture().onChanged { value in
                if value.translation.height > 10 { hideKeyboard() }
                showingSettings = false
                textFieldFocused = false
              }
            )
            .onTapGesture {
              showingSettings = false
              textFieldFocused = false
            }

          HStack {
            Button(action: {
              imagePickerSourceType = .photoLibrary
              isImagePickerPresented = true
            }) {
              Image(systemName: "photo.on.rectangle")
                .resizable()
                .scaledToFit()
                .frame(width: 24, height: 24)
            }
            .background(Color.clear)
            .cornerRadius(8)

            Button(action: {
              if UIImagePickerController.isSourceTypeAvailable(.camera) {
                imagePickerSourceType = .camera
                isImagePickerPresented = true
              } else {
                print("Camera not available")
              }
            }) {
              Image(systemName: "camera")
                .resizable()
                .scaledToFit()
                .frame(width: 24, height: 24)
            }
            .background(Color.clear)
            .cornerRadius(8)

            Button(action: {
              if viewModel.speechManager.isSpeaking { viewModel.speechManager.stop() }
              viewModel.speechRecognitionManager.toggleRecording()
            }) {
              MicButtonView(state: viewModel.speechRecognitionManager.recordingState)
            }
            .background(Color.clear)
            .cornerRadius(8)

            if viewModel.resourceManager.isModelValid && ModelType.fromPath(viewModel.resourceManager.modelPath) == .qwen3 {
              Button(action: {
                viewModel.thinkingMode.toggle()
                showThinkingModeNotification = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                  showThinkingModeNotification = false
                }
              }) {
                Image(systemName: "brain")
                  .resizable()
                  .scaledToFit()
                  .frame(width: 24, height: 24)
                  .foregroundColor(viewModel.thinkingMode ? .blue : .gray)
              }
              .background(Color.clear)
              .cornerRadius(8)
            }

            TextField(
              viewModel.resourceManager.isModelValid ? viewModel.resourceManager.isTokenizerValid ? "Prompt..." : "Select Tokenizer..." : "Select Model...",
              text: $viewModel.prompt,
              axis: .vertical
            )
            .padding(8)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(20)
            .lineLimit(1...10)
            .overlay(
              RoundedRectangle(cornerRadius: 20)
                .stroke(viewModel.isInputEnabled ? Color.blue : Color.gray, lineWidth: 1)
            )
            .disabled(!viewModel.isInputEnabled)
            .focused($textFieldFocused)
            .onAppear { textFieldFocused = false }
            .onTapGesture { showingSettings = false }
            .onChange(of: viewModel.prompt) { newValue in
              guard !newValue.isEmpty else { return }
              if let key = viewModel.currentPreloadKey(), key != viewModel.lastPreloadedKey {
                viewModel.lastPreloadedKey = key
                viewModel.loadModelIfNeededAsync()
              }
            }

            Button(action: viewModel.isGenerating ? viewModel.stop : viewModel.generate) {
              Image(systemName: viewModel.isGenerating ? "stop.circle" : "arrowshape.up.circle.fill")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(height: 28)
            }
            .disabled(viewModel.isGenerating ? viewModel.shouldStopGenerating : (!viewModel.isInputEnabled || viewModel.prompt.isEmpty || viewModel.speechRecognitionManager.recordingState == .transcribing))
          }
          .padding([.leading, .trailing, .bottom], 10)
        }
        .sheet(isPresented: $isImagePickerPresented, onDismiss: viewModel.addSelectedImageMessage) {
          ImagePicker(selectedImage: $viewModel.selectedImage, sourceType: imagePickerSourceType)
            .id(imagePickerSourceType.rawValue)
        }

        if showThinkingModeNotification {
          Text(viewModel.thinkingMode ? "Thinking mode enabled" : "Thinking mode disabled")
            .padding(8)
            .background(Color(UIColor.secondarySystemBackground))
            .cornerRadius(8)
            .transition(.opacity)
            .animation(.easeInOut(duration: 0.2), value: showThinkingModeNotification)
        }
      }
      .navigationBarTitle(
        viewModel.resourceManager.isModelValid ? viewModel.resourceManager.isTokenizerValid ? viewModel.resourceManager.modelName : "Select Tokenizer..." : "Select Model...",
        displayMode: .inline
      )
      .navigationBarItems(
        leading:
          Button(action: { showingSettings.toggle() }) {
            Image(systemName: "folder").imageScale(.large)
          },
        trailing:
          HStack {
            Menu {
              Section(header: Text("Memory")) {
                Text("Used: \(viewModel.resourceMonitor.usedMemory) Mb")
                Text("Available: \(viewModel.resourceMonitor.availableMemory) Mb")
              }
            } label: {
              Text("\(viewModel.resourceMonitor.usedMemory) Mb")
            }
            .onAppear { viewModel.resourceMonitor.start() }
            .onDisappear { viewModel.resourceMonitor.stop() }
            Button(action: { showingLogs = true }) {
              Image(systemName: "list.bullet.rectangle")
            }
            Button(action: { viewModel.autoSpeak.toggle() }) {
              Image(systemName: viewModel.autoSpeak ? "speaker.wave.2.fill" : "speaker.slash")
                .foregroundColor(viewModel.autoSpeak ? .blue : .primary)
            }
            Button(action: { viewModel.toggleVoiceClone() }) {
              VoiceCloneButtonView(
                isLoading: viewModel.isLoadingVoiceClone,
                isEnabled: viewModel.voiceCloneEnabled
              )
            }
          }
      )
      .sheet(isPresented: $showingLogs) {
        NavigationView {
          LogView(logManager: viewModel.logManager)
        }
      }
      .fileImporter(
        isPresented: Binding<Bool>(
          get: { pickerType != nil },
          set: { if !$0 { pickerType = nil } }
        ),
        allowedContentTypes: allowedContentTypes(),
        allowsMultipleSelection: false
      ) { [pickerType] result in
        viewModel.handleFileImportResult(pickerType, result) {
          showingSettings = false
          textFieldFocused = true
        }
      }
      .onAppear {
        viewModel.setup()
        viewModel.speechRecognitionManager.onTranscription = { transcribed in
          viewModel.prompt = transcribed
          textFieldFocused = false
          viewModel.generate()
        }
      }
    }
    .navigationViewStyle(StackNavigationViewStyle())
  }
  #endif

  // MARK: - Helpers

  private func allowedContentTypes() -> [UTType] {
    guard let pickerType else { return [] }
    switch pickerType {
    case .model:
      return [UTType(filenameExtension: "pte")].compactMap { $0 }
    case .tokenizer:
      return [UTType(filenameExtension: "bin"), UTType(filenameExtension: "model"), UTType(filenameExtension: "json")].compactMap { $0 }
    }
  }
}
