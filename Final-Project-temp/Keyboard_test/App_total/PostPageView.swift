import SwiftUI

struct PostPageView: View {
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Featured Section
                    FeaturedSection()
                        .padding(.top, 10)
                    
                    // Posts
                    ForEach(0..<5) { index in
                        ModernPostCard(
                            username: "User \(index + 1)",
                            timeAgo: "\(index + 1)h ago",
                            likes: Int.random(in: 50...500),
                            caption: "This is post caption \(index + 1) 😊"
                        )
                    }
                }
                .padding(.horizontal)
            }
            .navigationBarItems(
                leading: Text("EmotiMoji")
                    .font(.system(size: 28, weight: .bold)),
                trailing: NavigationLink(destination: MessagePage()) {
                    Image(systemName: "message")
                        .font(.title2)
                        .foregroundColor(.blue)
                }
            )
            .background(Color(.systemGray6).opacity(0.5))
        }
    }
}

struct FeaturedSection: View {
    let features = ["Trending", "Latest", "Following"]
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 15) {
                ForEach(features, id: \.self) { feature in
                    VStack(spacing: 8) {
                        RoundedRectangle(cornerRadius: 16)
                            .fill(LinearGradient(
                                colors: [.blue.opacity(0.2), .purple.opacity(0.2)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ))
                            .frame(width: 120, height: 160)
                            .overlay(
                                Text(feature)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.primary)
                            )
                    }
                }
            }
            .padding(.vertical, 5)
        }
    }
}

struct ModernPostCard: View {
    let username: String
    let timeAgo: String
    let likes: Int
    let caption: String
    @State private var isLiked = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Circle()
                    .fill(Color.blue.opacity(0.1))
                    .frame(width: 40, height: 40)
                    .overlay(
                        Text(String(username.prefix(1)))
                            .fontWeight(.semibold)
                            .foregroundColor(.blue)
                    )
                
                VStack(alignment: .leading) {
                    Text(username)
                        .font(.system(size: 16, weight: .semibold))
                    Text(timeAgo)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button(action: {}) {
                    Image(systemName: "ellipsis")
                        .foregroundColor(.secondary)
                }
            }
            
            // Content
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .frame(height: 300)
                .overlay(
                    Text("Post Content")
                        .foregroundColor(.secondary)
                )
                .shadow(color: Color.black.opacity(0.05), radius: 5, x: 0, y: 2)
            
            // Actions
            HStack(spacing: 20) {
                Button(action: { isLiked.toggle() }) {
                    HStack {
                        Image(systemName: isLiked ? "heart.fill" : "heart")
                            .foregroundColor(isLiked ? .red : .primary)
                        Text("\(likes)")
                            .font(.subheadline)
                    }
                }
                
                Button(action: {}) {
                    HStack {
                        Image(systemName: "bubble.right")
                        Text("Comment")
                            .font(.subheadline)
                    }
                }
                
                Spacer()
                
                Button(action: {}) {
                    Image(systemName: "bookmark")
                }
            }
            .foregroundColor(.primary)
            
            // Caption
            Text(caption)
                .font(.system(size: 14))
                .padding(.top, 4)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(20)
        .shadow(color: Color.black.opacity(0.05), radius: 10, x: 0, y: 5)
    }
}