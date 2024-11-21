import UIKit

class KeyboardViewController: UIInputViewController {
    
    private var emojiStackView: UIStackView!
    private var nextKeyboardButton: UIButton!
    private var containerView: UIView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupKeyboard()
        
        // 添加定时器来定期检查更新
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateRecommendedEmojis()
        }
    }
    
    private func setupKeyboard() {
        // 设置容器视图
        containerView = UIView(frame: view.bounds)
        containerView.backgroundColor = .systemBackground
        view.addSubview(containerView)
        
        // 设置切换键盘按钮
        nextKeyboardButton = UIButton(type: .system)
        nextKeyboardButton.setTitle(NSLocalizedString("🌐", comment: ""), for: [])
        nextKeyboardButton.sizeToFit()
        nextKeyboardButton.translatesAutoresizingMaskIntoConstraints = false
        nextKeyboardButton.addTarget(self, action: #selector(handleInputModeList(from:with:)), for: .allTouchEvents)
        containerView.addSubview(nextKeyboardButton)
        
        // 设置emoji栈视图
        emojiStackView = UIStackView()
        emojiStackView.axis = .horizontal
        emojiStackView.distribution = .fillEqually
        emojiStackView.spacing = 10
        emojiStackView.translatesAutoresizingMaskIntoConstraints = false
        containerView.addSubview(emojiStackView)
        
        // 设置约束
        NSLayoutConstraint.activate([
            // 容器视图约束
            containerView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            containerView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            containerView.topAnchor.constraint(equalTo: view.topAnchor),
            containerView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // 切换键盘按钮约束
            nextKeyboardButton.leadingAnchor.constraint(equalTo: containerView.leadingAnchor, constant: 10),
            nextKeyboardButton.bottomAnchor.constraint(equalTo: containerView.bottomAnchor, constant: -10),
            nextKeyboardButton.widthAnchor.constraint(equalToConstant: 40),
            nextKeyboardButton.heightAnchor.constraint(equalToConstant: 40),
            
            // Emoji栈视图约束
            emojiStackView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor, constant: 10),
            emojiStackView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor, constant: -10),
            emojiStackView.bottomAnchor.constraint(equalTo: nextKeyboardButton.topAnchor, constant: -10),
            emojiStackView.heightAnchor.constraint(equalToConstant: 50)
        ])
        
        // 设置默认emoji
        let defaultEmojis = ["😊", "😃", "😄", "🥳", "😁"]
        updateEmojiButtons(with: defaultEmojis)
    }
    
    private func updateRecommendedEmojis() {
        let userDefaults = UserDefaults(suiteName: "group.com.emotimoji.keyboard")
        if let emojis = userDefaults?.array(forKey: "RecommendedEmojis") as? [String] {
            DispatchQueue.main.async {
                self.updateEmojiButtons(with: emojis)
            }
        }
    }
    
    private func updateEmojiButtons(with emojis: [String]) {
        // 移除现有的emoji按钮
        emojiStackView.arrangedSubviews.forEach { $0.removeFromSuperview() }
        
        // 添加新的emoji按钮
        for emoji in emojis {
            let button = UIButton(type: .system)
            button.setTitle(emoji, for: .normal)
            button.titleLabel?.font = .systemFont(ofSize: 24)
            button.backgroundColor = .clear
            button.layer.cornerRadius = 8
            button.addTarget(self, action: #selector(emojiButtonTapped(_:)), for: .touchUpInside)
            
            // 添加触摸反馈
            button.addTarget(self, action: #selector(buttonTouchDown(_:)), for: .touchDown)
            button.addTarget(self, action: #selector(buttonTouchUp(_:)), for: [.touchUpInside, .touchUpOutside])
            
            emojiStackView.addArrangedSubview(button)
        }
    }
    
    @objc private func emojiButtonTapped(_ sender: UIButton) {
        if let emoji = sender.titleLabel?.text {
            let proxy = textDocumentProxy
            proxy.insertText(emoji)
            
            // 添加触觉反馈
            let generator = UIImpactFeedbackGenerator(style: .light)
            generator.impactOccurred()
        }
    }
    
    @objc private func buttonTouchDown(_ sender: UIButton) {
        UIView.animate(withDuration: 0.1) {
            sender.transform = CGAffineTransform(scaleX: 0.9, y: 0.9)
            sender.backgroundColor = UIColor.systemGray6
        }
    }
    
    @objc private func buttonTouchUp(_ sender: UIButton) {
        UIView.animate(withDuration: 0.1) {
            sender.transform = .identity
            sender.backgroundColor = .clear
        }
    }
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        nextKeyboardButton.isHidden = !needsInputModeSwitchKey
    }
    
    override func textWillChange(_ textInput: UITextInput?) {
        // 文本将要改变时的处理
    }
    
    override func textDidChange(_ textInput: UITextInput?) {
        // 文本已经改变时的处理
    }
}