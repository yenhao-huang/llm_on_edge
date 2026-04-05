/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI

struct MicButtonView: View {
  let state: SpeechRecognitionManager.RecordingState
  @State private var isAnimating = false

  var body: some View {
    ZStack {
      if case .recording = state {
        Circle()
          .stroke(Color.red.opacity(0.4), lineWidth: 2)
          .frame(width: 34, height: 34)
          .scaleEffect(isAnimating ? 1.4 : 1.0)
          .opacity(isAnimating ? 0 : 1)
          .animation(.easeOut(duration: 0.9).repeatForever(autoreverses: false), value: isAnimating)
          .onAppear { isAnimating = true }
          .onDisappear { isAnimating = false }
      }
      Image(systemName: iconName)
        .resizable()
        .scaledToFit()
        .frame(width: 24, height: 24)
        .foregroundColor(iconColor)
    }
    .frame(width: 34, height: 34)
  }

  private var iconName: String {
    switch state {
    case .recording:        return "mic.fill"
    case .transcribing:     return "waveform"
    case .loadingModel:     return "arrow.down.circle"
    case .permissionDenied: return "mic.slash"
    case .error:            return "exclamationmark.triangle"
    default:                return "mic"
    }
  }

  private var iconColor: Color {
    switch state {
    case .recording:    return .red
    case .transcribing: return .orange
    default:            return .primary
    }
  }
}

extension View {
  func hideKeyboard() {
    #if os(iOS)
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    #elseif os(macOS)
    NSApp.keyWindow?.makeFirstResponder(nil)
    #endif
  }
}
