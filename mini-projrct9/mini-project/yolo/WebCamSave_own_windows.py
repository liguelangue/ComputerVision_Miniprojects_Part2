import torch
import cv2


model = torch.hub.load('./yolov5', 'custom', path=r'D:\NEU\CS5330\mini_proj_9\yolov5\runs\train\exp18\weights\best.pt', source='local')  # 加载你的 best.pt 模型

# 初始化摄像头
cap = cv2.VideoCapture(0)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't load the video")
        break

    # YOLOv5 模型检测
    results = model(frame)

    # 可视化检测框和标签
    for det in results.pandas().xyxy[0].to_dict(orient="records"):
        x1, y1, x2, y2, conf, cls = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax']), det['confidence'], det['name']
        label = f"{cls} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('YOLOv5 Detection', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
