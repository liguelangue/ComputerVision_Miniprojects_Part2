import Foundation
import CoreML
import Vision
import UIKit

class ModelProcessor: ObservableObject {
   @Published var currentEmotion: String = "neutral"
   @Published var recommendedEmojis: [String] = []
   static let shared = ModelProcessor()
   
   // Shared UserDefaults for keyboard extension
   private let userDefaults = UserDefaults(suiteName: "group.com.emotimoji.keyboard")
   
   // Current selected emoji collection
   private var currentEmojiCollection: [String: [String]] = [:]
   
   // Update emoji collection from EmojiManagerPage
   func updateEmojiCollection(_ collection: [String: [String]]) {
       currentEmojiCollection = collection
       saveEmojiCollectionToUserDefaults()
   }
   
   // Process image with ML model
   func processImage(_ image: UIImage) {
       guard let ciImage = CIImage(image: image) else { return }
       
       let config = MLModelConfiguration()
       guard let model = try? MobileNetV3(configuration: config) else { return }
       
       let request = VNCoreMLRequest(model: try! VNCoreMLModel(for: model.model)) { [weak self] request, error in
           guard let results = request.results as? [VNClassificationObservation],
                 let topResult = results.first else { return }
           
           // Map model output to emotion category
           let emotion = self?.mapOutputToEmotion(topResult.identifier)
           DispatchQueue.main.async {
               self?.updateEmotionAndEmojis(emotion ?? "neutral")
           }
       }
       
       try? VNImageRequestHandler(ciImage: ciImage).perform([request])
   }
   
   // Map model output to emotion categories
   private func mapOutputToEmotion(_ output: String) -> String {
       // Implement mapping logic based on your model's output
       // This is a placeholder implementation
       return output
   }
   
   // Update emotion and recommended emojis
   private func updateEmotionAndEmojis(_ emotion: String) {
       currentEmotion = emotion
       recommendedEmojis = currentEmojiCollection[emotion] ?? []
       
       // Update keyboard extension
       userDefaults?.set(recommendedEmojis, forKey: "RecommendedEmojis")
       userDefaults?.synchronize()
       
       // Notify CameraView
       NotificationCenter.default.post(
           name: NSNotification.Name("EmotionDetected"),
           object: nil,
           userInfo: ["emotion": emotion, "emojis": recommendedEmojis]
       )
   }
   
   private func saveEmojiCollectionToUserDefaults() {
       userDefaults?.set(currentEmojiCollection, forKey: "CurrentEmojiCollection")
       userDefaults?.synchronize()
   }
}
