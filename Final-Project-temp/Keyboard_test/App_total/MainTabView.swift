import SwiftUI

struct MainTabView: View {
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            PostPageView()
                .tabItem {
                    Image(systemName: selectedTab == 0 ? "house.fill" : "house")
                    Text("Home")
                }
                .tag(0)
            
            HomePage()
                .tabItem {
                    Image(systemName: selectedTab == 1 ? "camera.fill" : "camera")
                    Text("Detect")
                }
                .tag(1)
            
            EmojiManagerPageView()
                .tabItem {
                    Image(systemName: selectedTab == 2 ? "face.smiling.fill" : "face.smiling")
                    Text("Emojis")
                }
                .tag(2)
            
            UserPageView()
                .tabItem {
                    Image(systemName: selectedTab == 3 ? "person.fill" : "person")
                    Text("Profile")
                }
                .tag(3)
        }
        .accentColor(.blue)
    }
}