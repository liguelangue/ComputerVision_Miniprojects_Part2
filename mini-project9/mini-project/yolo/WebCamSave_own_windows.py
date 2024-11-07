import torch
import cv2


model = torch.hub.load('./yolov5', 'custom', path=r'yolov5\runs\train\exp18\weights\best.pt', source='local')

cap = cv2.VideoCapture(0)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't load the video")
        break


    results = model(frame)


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
