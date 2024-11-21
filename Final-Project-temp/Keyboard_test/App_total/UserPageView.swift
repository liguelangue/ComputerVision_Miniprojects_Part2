import SwiftUI

struct UserPageView: View {
    @State private var selectedTab = "posts"
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 25) {
                    // Profile Header Card
                    ProfileHeaderCard()
                    
                    // Stats Card
                    StatsCard()
                    
                    // Content Tabs
                    ContentTabSelector(selectedTab: $selectedTab)
                    
                    // Grid Content
                    if selectedTab == "posts" {
                        PostsGrid()
                    } else {
                        SavedEmojisGrid()
                    }
                }
                .padding()
            }
            .navigationBarItems(
                leading: Text("Profile")
                    .font(.system(size: 28, weight: .bold)),
                trailing: Button(action: {}) {
                    Image(systemName: "gearshape")
                        .font(.title2)
                        .foregroundColor(.primary)
                }
            )
            .background(Color(.systemGray6).opacity(0.5))
        }
    }
}

struct ProfileHeaderCard: View {
    var body: some View {
        VStack(spacing: 20) {
            // Profile Image
            ZStack {
                Circle()
                    .fill(LinearGradient(
                        colors: [.blue.opacity(0.2), .purple.opacity(0.2)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ))
                    .frame(width: 100, height: 100)
                
                Text("AT")
                    .font(.title)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
            }
            
            // Profile Info
            VStack(spacing: 8) {
                Text("Anning Tian")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Text("Emoji enthusiast | Developer")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Link("emotimoji.dev", destination: URL(string: "https://emotimoji.dev")!)
                    .font(.subheadline)
                    .foregroundColor(.blue)
            }
            
            // Edit Profile Button
            Button(action: {}) {
                Text("Edit Profile")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.primary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(Color(.systemBackground))
                    .cornerRadius(12)
                    .shadow(color: Color.black.opacity(0.05), radius: 5)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(20)
        .shadow(color: Color.black.opacity(0.05), radius: 10)
    }
}

struct StatsCard: View {
    var body: some View {
        HStack {
            StatItem(value: "256", title: "Posts")
            Divider()
            StatItem(value: "12.8K", title: "Followers")
            Divider()
            StatItem(value: "1.2K", title: "Following")
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(20)
        .shadow(color: Color.black.opacity(0.05), radius: 10)
    }
}

struct StatItem: View {
    let value: String
    let title: String
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.system(size: 20, weight: .bold))
            Text(title)
                .font(.system(size: 14))
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct ContentTabSelector: View {
    @Binding var selectedTab: String
    
    var body: some View {
        HStack(spacing: 0) {
            TabButton(
                title: "Posts",
                systemImage: "square.grid.2x2",
                isSelected: selectedTab == "posts",
                action: { selectedTab = "posts" }
            )
            
            TabButton(
                title: "Saved Emojis",
                systemImage: "heart",
                isSelected: selectedTab == "saved",
                action: { selectedTab = "saved" }
            )
        }
        .padding(6)
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(color: Color.black.opacity(0.05), radius: 5)
    }
}

struct TabButton: View {
    let title: String
    let systemImage: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: systemImage)
                    .font(.system(size: 20))
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(isSelected ? Color.blue.opacity(0.1) : Color.clear)
            .cornerRadius(12)
            .foregroundColor(isSelected ? .blue : .primary)
        }
    }
}

struct PostsGrid: View {
    let columns = Array(repeating: GridItem(.flexible(), spacing: 2), count: 3)
    
    var body: some View {
        LazyVGrid(columns: columns, spacing: 2) {
            ForEach(0..<15) { _ in
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.systemBackground))
                    .aspectRatio(1, contentMode: .fit)
                    .overlay(
                        Image(systemName: "photo")
                            .font(.title)
                            .foregroundColor(.secondary)
                    )
            }
        }
    }
}

struct SavedEmojisGrid: View {
    let columns = Array(repeating: GridItem(.flexible(), spacing: 15), count: 4)
    
    var body: some View {
        LazyVGrid(columns: columns, spacing: 15) {
            ForEach(0..<20) { _ in
                Text("😊")
                    .font(.system(size: 40))
                    .frame(width: 70, height: 70)
                    .background(Color(.systemBackground))
                    .cornerRadius(12)
                    .shadow(color: Color.black.opacity(0.05), radius: 5)
            }
        }
    }
}