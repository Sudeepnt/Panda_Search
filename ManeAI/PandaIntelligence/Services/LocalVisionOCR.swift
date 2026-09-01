import AppKit
import AVFoundation
import Foundation
import Vision

/// Produces local searchable image context with Apple's Vision framework.
/// Text recognition covers screenshots/documents, while visual labels make
/// ordinary photos discoverable without sending them off the Mac.
enum LocalVisionOCR {
    static func recognizeText(in url: URL) async -> String {
        guard let image = NSImage(contentsOf: url),
              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return ""
        }

        return await recognize(cgImage: cgImage)
    }

    /// Samples representative frames so videos receive the same local OCR and
    /// visual-label treatment as images. The returned text is sent alongside
    /// the original video path to the persistent local vector index.
    static func recognizeVideo(in url: URL) async -> String {
        let asset = AVURLAsset(url: url)
        let seconds = max(CMTimeGetSeconds(asset.duration), 1)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1024, height: 1024)

        let fractions: [Double] = seconds < 8 ? [0.2, 0.6] : [0.1, 0.5, 0.9]
        var descriptions: [String] = []
        for (index, fraction) in fractions.enumerated() {
            let time = CMTime(seconds: seconds * fraction, preferredTimescale: 600)
            guard let frame = try? generator.copyCGImage(at: time, actualTime: nil) else { continue }
            let description = await recognize(cgImage: frame)
            if !description.isEmpty {
                descriptions.append("[video frame \(index + 1)]\n\(description)")
            }
        }
        return descriptions.joined(separator: "\n\n")
    }

    private static func recognize(cgImage: CGImage) async -> String {
        return await withCheckedContinuation { continuation in
            let textRequest = VNRecognizeTextRequest()
            textRequest.recognitionLevel = .accurate
            textRequest.usesLanguageCorrection = true
            let classificationRequest = VNClassifyImageRequest()

            DispatchQueue.global(qos: .userInitiated).async {
                try? VNImageRequestHandler(cgImage: cgImage, options: [:]).perform([
                    textRequest,
                    classificationRequest
                ])

                let lines = (textRequest.results as? [VNRecognizedTextObservation])?
                    .compactMap { $0.topCandidates(1).first?.string }
                    ?? []
                let labels = (classificationRequest.results as? [VNClassificationObservation])?
                    .filter { $0.confidence >= 0.25 }
                    .prefix(12)
                    .map(\.identifier)
                    ?? []
                let recognizedText = lines.isEmpty ? "" : "[recognized text in image]\n" + lines.joined(separator: "\n")
                let visualLabels = labels.isEmpty ? "" : "[local visual labels]\n" + labels.joined(separator: ", ")
                continuation.resume(returning: [recognizedText, visualLabels]
                    .filter { !$0.isEmpty }
                    .joined(separator: "\n\n"))
            }
        }
    }
}
