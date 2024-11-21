import SwiftUI
import AVFoundation
import PhotosUI
import AVKit

struct HomePage: View {
    @State private var isCameraActive = false
    @State private var showCameraPermissionAlert = false
    @State private var selectedImage: UIImage?
    @State private var selectedVideoURL: URL?
    @State private var showImagePicker = false
    @State private var showVideoPicker = false
    @State private var showProcessingView = false
    
    // Test directories
    let testImagePath = "/Users/anningtian/Desktop/test_images"
    let testVideoPath = "/Users/anningtian/Desktop/test_videos"
    
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
                
                VStack(spacing: 25) {
                    // Header
                    HStack {
                        Text("EmotiMoji")
                            .font(.system(size: 28, weight: .bold))
                            .foregroundColor(.primary)
                        Spacer()
                        NavigationLink(destination: MessagePage()) {
                            Image(systemName: "message")
                                .font(.title2)
                        }
                    }
                    .padding()
                    
                    Spacer()
                    
                    // Three detection options
                    VStack(spacing: 25) {
                        // Live Camera Detection
                        DetectionOptionCard(
                            title: "Live Camera",
                            description: "Real-time emotion detection",
                            iconName: "camera.fill",
                            color: .blue
                        ) {
                            checkAndRequestCameraPermission()
                        }
                        
                        // Image Upload
                        DetectionOptionCard(
                            title: "Upload Image",
                            description: "Detect from photo",
                            iconName: "photo.fill",
                            color: .purple
                        ) {
                            showImagePicker = true
                        }
                        
                        // Video Upload
                        DetectionOptionCard(
                            title: "Upload Video",
                            description: "Detect from video",
                            iconName: "video.fill",
                            color: .orange
                        ) {
                            showVideoPicker = true
                        }
                    }
                    .padding(.horizontal)
                    
                    Spacer()
                    
                    // Status text
                    if showProcessingView {
                        ProcessingView()
                    }
                }
            }
            .sheet(isPresented: $showImagePicker) {
                ImagePicker(selectedImage: $selectedImage)
            }
            .sheet(isPresented: $showVideoPicker) {
                VideoPicker(selectedVideoURL: $selectedVideoURL)
            }
            .alert("Camera Permission Required", isPresented: $showCameraPermissionAlert) {
                Button("Go to Settings", role: .none) {
                    if let settingsURL = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(settingsURL)
                    }
                }
                Button("Cancel", role: .cancel) {}
            }
        }
    }
    
    private func checkAndRequestCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            toggleCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    toggleCamera()
                }
            }
        default:
            showCameraPermissionAlert = true
        }
    }
    
    private func toggleCamera() {
        withAnimation {
            isCameraActive.toggle()
            showProcessingView = isCameraActive
        }
    }
}

// Helper Views
struct DetectionOptionCard: View {
    let title: String
    let description: String
    let iconName: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 15) {
                Image(systemName: iconName)
                    .font(.title)
                    .foregroundColor(.white)
                    .frame(width: 50, height: 50)
                    .background(color)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.headline)
                    Text(description)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Image(systemName: "chevron.right")
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(16)
            .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
        }
    }
}

struct ProcessingView: View {
    @State private var isAnimating = false
    
    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(Color.blue)
                .frame(width: 10, height: 10)
                .opacity(isAnimating ? 1 : 0.3)
                .animation(Animation.easeInOut(duration: 0.5).repeatForever().delay(0), value: isAnimating)
            
            Text("Processing...")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(.vertical)
        .onAppear {
            isAnimating = true
        }
    }
}

// Image Picker
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.presentationMode.wrappedValue.dismiss()
            
            guard let provider = results.first?.itemProvider else { return }
            
            if provider.canLoadObject(ofClass: UIImage.self) {
                provider.loadObject(ofClass: UIImage.self) { image, _ in
                    DispatchQueue.main.async {
                        self.parent.selectedImage = image as? UIImage
                    }
                }
            }
        }
    }
}

// Video Picker
struct VideoPicker: UIViewControllerRepresentable {
    @Binding var selectedVideoURL: URL?
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.movie])
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let parent: VideoPicker
        
        init(_ parent: VideoPicker) {
            self.parent = parent
        }
        
        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            parent.selectedVideoURL = url
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}