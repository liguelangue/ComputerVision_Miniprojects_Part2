import SwiftUI

struct MessagePage: View {
    @State private var searchText = ""
    @State private var selectedFilter = "All"
    let filters = ["All", "Unread", "Requests"]
    
    var body: some View {
        VStack(spacing: 0) {
            // Modern Search Bar
            ModernSearchBar(text: $searchText)
                .padding()
            
            // Filters
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 15) {
                    ForEach(filters, id: \.self) { filter in
                        FilterChip(
                            title: filter,
                            isSelected: selectedFilter == filter,
                            action: { selectedFilter = filter }
                        )
                    }
                }
                .padding(.horizontal)
            }
            .padding(.vertical, 8)
            
            // Messages List
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(0..<10) { index in
                        ModernMessageRow(
                            username: "User \(index + 1)",
                            lastMessage: "Last message preview text goes here...",
                            time: "\(index + 1)h",
                            hasUnread: index % 3 == 0
                        )
                    }
                }
                .padding()
            }
        }
        .navigationBarItems(
            leading: Text("Messages")
                .font(.system(size: 28, weight: .bold)),
            trailing: Button(action: {}) {
                Image(systemName: "square.and.pencil")
                    .font(.title2)
                    .foregroundColor(.blue)
            }
        )
        .background(Color(.systemGray6).opacity(0.5))
    }
}

struct ModernSearchBar: View {
    @Binding var text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.secondary)
            
            TextField("Search messages", text: $text)
                .textFieldStyle(PlainTextFieldStyle())
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(color: Color.black.opacity(0.05), radius: 5)
    }
}

struct FilterChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 14, weight: .medium))
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(isSelected ? Color.blue : Color(.systemBackground))
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(20)
                .shadow(color: Color.black.opacity(0.05), radius: 5)
        }
    }
}

struct ModernMessageRow: View {
    let username: String
    let lastMessage: String
    let time: String
    let hasUnread: Bool
    
    var body: some View {
        HStack(spacing: 12) {
            // Avatar
            ZStack {
                Circle()
                    .fill(LinearGradient(
                        colors: [.blue.opacity(0.2), .purple.opacity(0.2)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ))
                    .frame(width: 50, height: 50)
                
                Text(String(username.prefix(1)))
                    .font(.system(size: 18, weight: .medium))
                    .foregroundColor(.primary)
            }
            
            // Message Content
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(username)
                        .font(.system(size: 16, weight: .semibold))
                    Spacer()
                    Text(time)
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                }
                
                Text(lastMessage)
                    .font(.system(size: 14))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
            
            if hasUnread {
                Circle()
                    .fill(Color.blue)
                    .frame(width: 8, height: 8)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.05), radius: 5)
    }
}