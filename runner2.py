import cv2
import pyautogui
from ultralytics import YOLO

# 1. SPEED HACK: Remove built-in delay
pyautogui.PAUSE = 0 
pyautogui.FAILSAFE = True

def start_auto_driver():
    # Load the NEW 320px model you just created
    model = YOLO('yolov8n.onnx')
    
    # 2. CAMERA FIX: Switched back to 0 (Standard default)
    # If 0 fails, try 2. (1 is usually reserved for internal metadata on Linux)
    cap = cv2.VideoCapture(0)
    
    # Set Camera Resolution
    W, H = 320, 240
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    
    if not cap.isOpened():
        print("ERROR: Camera 0 not found. Try changing cv2.VideoCapture(0) to 1 or 2.")
        return

    center_x = W // 2
    print(f"AI Driver Initialized at {W}x{H}. Press 'q' to quit.")

    frame_count = 0
    last_action = "Idling"
    last_box = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        # 3. LOGIC: Skip frames to save CPU
        if frame_count % 4 == 0:
            # We match the 'imgsz=320' to the ONNX model we created
            results = model(frame, stream=True, classes=[2], conf=0.4, verbose=False, imgsz=320)
            
            closest_car = None
            max_area = 0

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        closest_car = (x1, y1, x2, y2)
            
            last_box = closest_car
        
        # --- CONTROL LOGIC ---
        if last_box:
            x1, y1, x2, y2 = last_box
            
            # Draw visual feedback
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame, (center_x, 0), (center_x, H), (255, 255, 255), 1)

            car_center_x = x1 + (x2 - x1) // 2
            
            # STEERING
            if car_center_x < (center_x - 30): 
                pyautogui.keyDown('a')
                pyautogui.keyUp('d')
                last_action = "Turning LEFT"
            elif car_center_x > (center_x + 30):
                pyautogui.keyDown('d')
                pyautogui.keyUp('a')
                last_action = "Turning RIGHT"
            else:
                pyautogui.keyUp('a')
                pyautogui.keyUp('d')
                last_action = "Centered"

            # SPEED
            box_width = x2 - x1
            if box_width < 70: 
                pyautogui.keyDown('w')
                pyautogui.keyUp('s')
                last_action += " + GAS"
            elif box_width > 150:
                pyautogui.keyDown('s')
                pyautogui.keyUp('w')
                last_action += " + BRAKE"
            else:
                pyautogui.keyUp('w')
                pyautogui.keyUp('s')
        else:
            pyautogui.keyUp('w')
            pyautogui.keyUp('s')
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')
            last_action = "Scanning..."

        cv2.putText(frame, last_action, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('AI Driver View', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_auto_driver()