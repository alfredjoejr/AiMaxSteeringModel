import cv2
from ultralytics import YOLO

def start_yolo_detection():
    # Load the "Nano" model. It's the smallest and fastest version.
    # It will auto-download 'yolov8n.pt' the first time you run this.
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Camera error")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        # stream=True makes it run faster for video
        # classes=[2] tells it to ONLY look for cars (class ID 2 is car in COCO dataset)
        results = model(frame, stream=True, classes=[2], conf=0.5)

        # Iterate through results to draw boxes manually (or use results[0].plot() for auto-drawing)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw simple box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, "Car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.imshow('YOLOv8 Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_yolo_detection()