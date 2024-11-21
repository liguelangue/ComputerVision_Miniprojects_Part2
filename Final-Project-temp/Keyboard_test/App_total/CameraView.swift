import SwiftUI

struct CameraView: View {
   @StateObject private var modelProcessor = ModelProcessor.shared
   @State private var detectedEmotion: String = ""
   @State private var recommendedEmojis: [String] = []
   
   var body: some View {
       VStack {
           Text("Detected Emotion: \(detectedEmotion)")
               .font(.headline)
           
           ScrollView(.horizontal, showsIndicators: false) {
               HStack(spacing: 10) {
                   ForEach(recommendedEmojis, id: \.self) { emoji in
                       Text(emoji)
                           .font(.system(size: 32))
                   }
               }
               .padding()
           }
       }
       .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("EmotionDetected"))) { notification in
           if let emotion = notification.userInfo?["emotion"] as? String,
              let emojis = notification.userInfo?["emojis"] as? [String] {
               detectedEmotion = emotion
               recommendedEmojis = emojis
               updateKeyboardEmojis(emojis)
           }
       }
   }
   
   private func updateKeyboardEmojis(_ emojis: [String]) {
       let userDefaults = UserDefaults(suiteName: "group.com.emotimoji.keyboard")
       userDefaults?.set(emojis, forKey: "RecommendedEmojis")
       userDefaults?.synchronize()
   }
}
