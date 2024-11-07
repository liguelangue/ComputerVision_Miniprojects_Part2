import torch
import cv2

# load trained model under yolov5 root file
model = torch.hub.load('./yolov5', 'custom', path='yolov5/runs/train/exp2/weights/best.pt', source='local')  # 加载你的 best.pt 模型

# initial live cam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧")
        break

    # YOLOv5 detection
    results = model(frame)

    # visualization of detected object 
    for det in results.pandas().xyxy[0].to_dict(orient="records"):
        x1, y1, x2, y2, conf, cls = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax']), det['confidence'], det['name']
        label = f"{cls} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv5 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()