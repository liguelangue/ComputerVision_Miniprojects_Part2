import SwiftUI

struct EmojiManagerPageView: View {
    @State private var selectedCollection = "iOS Emojis"
    @State private var searchText = ""
    
    let emojiCollections = [
        "iOS Emojis": [
            "Frequently Used": ["😊", "😂", "❤️", "🎉", "👍", "✨"],
            "Emotions": ["😊", "😃", "😄", "🥳", "😁", "😢", "😭", "😔"],
            "Animals": ["🐶", "🐱", "🐼", "🦁", "🐯", "🦊", "🦝"],
            "Nature": ["🌸", "🌺", "🌷", "🌹", "🌻", "🌼", "🌿"],
            "Food": ["🍎", "🍕", "🍣", "🍜", "🍪", "🍩", "🍰"]
        ],
        "Custom Packs": [
            "Favorites": ["🌟", "💫", "⭐️"],
            "Custom Pack 1": ["🤖", "👾", "👽"],
            "Coming Soon": ["✨"]
        ]
    ]
    
    var body: some View {
        NavigationView {
            ZStack {
                // Background gradient
                LinearGradient(
                    gradient: Gradient(colors: [Color.blue.opacity(0.1), Color.purple.opacity(0.1)]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                VStack(spacing: 0) {
                    // Search Bar
                    ModernSearchBar(text: $searchText)
                        .padding()
                    
                    // Collection Selector
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 20) {
                            ForEach(["iOS Emojis", "Custom Packs"], id: \.self) { collection in
                                ModernCollectionCard(
                                    title: collection,
                                    icon: collection == "iOS Emojis" ? "face.smiling" : "star",
                                    isSelected: selectedCollection == collection,
                                    action: { selectedCollection = collection }
                                )
                            }
                        }
                        .padding()
                    }
                    
                    // Emoji Categories
                    ScrollView {
                        VStack(spacing: 20) {
                            ForEach(Array(emojiCollections[selectedCollection]?.keys ?? [:].keys), id: \.self) { category in
                                ModernEmojiSection(
                                    title: category,
                                    emojis: emojiCollections[selectedCollection]?[category] ?? []
                                )
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Emoji Manager")
            .navigationBarItems(
                trailing: Menu {
                    Button("Create Custom Pack", action: {})
                    Button("Manage Packs", action: {})
                    Button("Settings", action: {})
                } label: {
                    Image(systemName: "plus.circle")
                        .font(.title2)
                        .foregroundColor(.blue)
                }
            )
           .onChange(of: selectedCollection) { newValue in
               if let collection = emojiCollections[newValue] {
                   ModelProcessor.shared.updateEmojiCollection(collection)
               }
           }
        }
    }
}

struct ModernCollectionCard: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(isSelected ? .blue : .primary)
                Text(title)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(isSelected ? .blue : .primary)
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(12)
            .shadow(color: Color.black.opacity(0.05), radius: 5)
        }
    }
}

struct ModernEmojiSection: View {
    let title: String
    let emojis: [String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.system(size: 18, weight: .semibold))
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 6), spacing: 12) {
                ForEach(emojis, id: \.self) { emoji in
                    Text(emoji)
                        .font(.system(size: 32))
                        .frame(width: 50, height: 50)
                        .background(Color(.systemBackground))
                        .cornerRadius(12)
                        .shadow(color: Color.black.opacity(0.05), radius: 5)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.05), radius: 10)
    }
}
