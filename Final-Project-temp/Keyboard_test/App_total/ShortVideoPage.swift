import SwiftUI
import AVKit

struct ShortVideoPage: View {
    // 示例视频链接列表
    let videos: [String] = [
        "/Users/anningtian/Downloads/he5ab9b67_1187363.mov",
        "https://example.com/video2.mp4",
        "https://example.com/video3.mp4"
    ]

    var body: some View {
        GeometryReader { geometry in
            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 0) {
                    ForEach(videos, id: \.self) { videoURL in
                        VideoPlayerView(videoURL: videoURL)
                            .frame(width: geometry.size.width, height: geometry.size.height)
                            .clipped()
                    }
                }
            }
        }
        .navigationBarTitle("Short Videos", displayMode: .inline)
        .edgesIgnoringSafeArea(.all)
    }
}

struct VideoPlayerView: View {
    var videoURL: String

    var body: some View {
        // 这里仅展示视频URL，实际项目中应使用AVPlayer来播放视频
        ZStack {
            Color.black
            Text(videoURL)
                .foregroundColor(.white)
                .bold()
        }
    }
}
