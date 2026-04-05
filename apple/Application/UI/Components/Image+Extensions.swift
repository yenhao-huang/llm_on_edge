/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorchLLM
import SwiftUI

#if os(iOS)
import UIKit

extension UIImage {
  func centerCropped(to sideSize: CGFloat) -> UIImage {
    precondition(sideSize > 0)
    let format = UIGraphicsImageRendererFormat.default()
    format.scale = 1
    format.opaque = false
    return UIGraphicsImageRenderer(size: CGSize(width: sideSize, height: sideSize), format: format).image { _ in
      let scaleFactor = max(sideSize / size.width, sideSize / size.height)
      let scaledWidth = size.width * scaleFactor
      let scaledHeight = size.height * scaleFactor
      let originX = (sideSize - scaledWidth) / 2
      let originY = (sideSize - scaledHeight) / 2
      draw(in: CGRect(x: originX, y: originY, width: scaledWidth, height: scaledHeight))
    }
  }

  func rgbBytes() -> [UInt8]? {
    guard let cgImage = cgImage else { return nil }
    let pixelWidth = Int(cgImage.width)
    let pixelHeight = Int(cgImage.height)
    let pixelCount = pixelWidth * pixelHeight
    let bytesPerPixel = 4
    let bytesPerRow = pixelWidth * bytesPerPixel
    var rgbaBuffer = [UInt8](repeating: 0, count: pixelCount * bytesPerPixel)
    guard let context = CGContext(
      data: &rgbaBuffer,
      width: pixelWidth,
      height: pixelHeight,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else { return nil }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))
    var rgbBytes = [UInt8](repeating: 0, count: pixelCount * 3)
    for pixelIndex in 0..<pixelCount {
      let sourceIndex = pixelIndex * bytesPerPixel
      rgbBytes[pixelIndex] = rgbaBuffer[sourceIndex + 0]
      rgbBytes[pixelIndex + pixelCount] = rgbaBuffer[sourceIndex + 1]
      rgbBytes[pixelIndex + 2 * pixelCount] = rgbaBuffer[sourceIndex + 2]
    }
    return rgbBytes
  }

  func rgbBytesNormalized(mean: [Float] = [0, 0, 0], std: [Float] = [1, 1, 1]) -> [Float]? {
    precondition(mean.count == 3 && std.count == 3)
    precondition(std[0] != 0 && std[1] != 0 && std[2] != 0)
    guard let rgbBytes = rgbBytes() else { return nil }
    let pixelCount = rgbBytes.count / 3
    var rgbBytesNormalized = [Float](repeating: 0, count: pixelCount * 3)
    for pixelIndex in 0..<pixelCount {
      rgbBytesNormalized[pixelIndex] = (Float(rgbBytes[pixelIndex]) / 255.0 - mean[0]) / std[0]
      rgbBytesNormalized[pixelIndex + pixelCount] = (Float(rgbBytes[pixelIndex + pixelCount]) / 255.0 - mean[1]) / std[1]
      rgbBytesNormalized[pixelIndex + 2 * pixelCount] = (Float(rgbBytes[pixelIndex + 2 * pixelCount]) / 255.0 - mean[2]) / std[2]
    }
    return rgbBytesNormalized
  }

  func asImage(_ sideSize: CGFloat) -> ExecuTorchLLM.Image {
    return Image(
      data: Data(centerCropped(to: sideSize).rgbBytes() ?? []),
      width: Int(sideSize),
      height: Int(sideSize),
      channels: 3
    )
  }

  func asNormalizedImage(_ sideSize: CGFloat, mean: [Float] = [0.485, 0.456, 0.406], std: [Float] = [0.229, 0.224, 0.225]) -> ExecuTorchLLM.Image {
    return Image(
      float: (centerCropped(to: sideSize).rgbBytesNormalized(mean: mean, std: std) ?? []).withUnsafeBufferPointer { Data(buffer: $0) },
      width: Int(sideSize),
      height: Int(sideSize),
      channels: 3
    )
  }
}

#elseif os(macOS)
import AppKit

extension NSImage {
  func centerCropped(to sideSize: CGFloat) -> NSImage {
    precondition(sideSize > 0)
    let newImage = NSImage(size: NSSize(width: sideSize, height: sideSize))
    newImage.lockFocus()
    let scaleFactor = max(sideSize / size.width, sideSize / size.height)
    let scaledWidth = size.width * scaleFactor
    let scaledHeight = size.height * scaleFactor
    let originX = (sideSize - scaledWidth) / 2
    let originY = (sideSize - scaledHeight) / 2
    draw(in: NSRect(x: originX, y: originY, width: scaledWidth, height: scaledHeight),
         from: NSRect(origin: .zero, size: size),
         operation: .copy,
         fraction: 1.0)
    newImage.unlockFocus()
    return newImage
  }

  func rgbBytes() -> [UInt8]? {
    guard let cgImage = cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
    let pixelWidth = Int(cgImage.width)
    let pixelHeight = Int(cgImage.height)
    let pixelCount = pixelWidth * pixelHeight
    let bytesPerPixel = 4
    let bytesPerRow = pixelWidth * bytesPerPixel
    var rgbaBuffer = [UInt8](repeating: 0, count: pixelCount * bytesPerPixel)
    guard let context = CGContext(
      data: &rgbaBuffer,
      width: pixelWidth,
      height: pixelHeight,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else { return nil }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))
    var rgbBytes = [UInt8](repeating: 0, count: pixelCount * 3)
    for pixelIndex in 0..<pixelCount {
      let sourceIndex = pixelIndex * bytesPerPixel
      rgbBytes[pixelIndex] = rgbaBuffer[sourceIndex + 0]
      rgbBytes[pixelIndex + pixelCount] = rgbaBuffer[sourceIndex + 1]
      rgbBytes[pixelIndex + 2 * pixelCount] = rgbaBuffer[sourceIndex + 2]
    }
    return rgbBytes
  }

  func rgbBytesNormalized(mean: [Float] = [0, 0, 0], std: [Float] = [1, 1, 1]) -> [Float]? {
    precondition(mean.count == 3 && std.count == 3)
    precondition(std[0] != 0 && std[1] != 0 && std[2] != 0)
    guard let rgbBytes = rgbBytes() else { return nil }
    let pixelCount = rgbBytes.count / 3
    var rgbBytesNormalized = [Float](repeating: 0, count: pixelCount * 3)
    for pixelIndex in 0..<pixelCount {
      rgbBytesNormalized[pixelIndex] = (Float(rgbBytes[pixelIndex]) / 255.0 - mean[0]) / std[0]
      rgbBytesNormalized[pixelIndex + pixelCount] = (Float(rgbBytes[pixelIndex + pixelCount]) / 255.0 - mean[1]) / std[1]
      rgbBytesNormalized[pixelIndex + 2 * pixelCount] = (Float(rgbBytes[pixelIndex + 2 * pixelCount]) / 255.0 - mean[2]) / std[2]
    }
    return rgbBytesNormalized
  }

  func asImage(_ sideSize: CGFloat) -> ExecuTorchLLM.Image {
    return Image(
      data: Data(centerCropped(to: sideSize).rgbBytes() ?? []),
      width: Int(sideSize),
      height: Int(sideSize),
      channels: 3
    )
  }

  func asNormalizedImage(_ sideSize: CGFloat, mean: [Float] = [0.485, 0.456, 0.406], std: [Float] = [0.229, 0.224, 0.225]) -> ExecuTorchLLM.Image {
    return Image(
      float: (centerCropped(to: sideSize).rgbBytesNormalized(mean: mean, std: std) ?? []).withUnsafeBufferPointer { Data(buffer: $0) },
      width: Int(sideSize),
      height: Int(sideSize),
      channels: 3
    )
  }
}
#endif
