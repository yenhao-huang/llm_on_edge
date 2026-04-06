import SwiftUI

struct VoiceCloneButtonView: View {
  let isLoading: Bool
  let isEnabled: Bool
  @State private var isPulsing = false

  var body: some View {
    ZStack {
      if isEnabled && !isLoading {
        Circle()
          .stroke(Color.purple.opacity(0.35), lineWidth: 2)
          .frame(width: 34, height: 34)
          .scaleEffect(isPulsing ? 1.4 : 1.0)
          .opacity(isPulsing ? 0 : 1)
          .animation(.easeOut(duration: 1.1).repeatForever(autoreverses: false), value: isPulsing)
          .onAppear { isPulsing = true }
          .onDisappear { isPulsing = false }
      }

      if isLoading {
        ProgressView()
          .frame(width: 24, height: 24)
      } else {
        Image(systemName: isEnabled ? "person.crop.circle.fill.badge.checkmark" : "person.crop.circle")
          .resizable()
          .scaledToFit()
          .frame(width: 24, height: 24)
          .foregroundColor(isEnabled ? .purple : .primary)
      }
    }
    .frame(width: 34, height: 34)
  }
}
