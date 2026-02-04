import cv2
import pyautogui
from ultralytics import YOLO

# SAFETY FAILSAFE: Move mouse to upper-left corner to kill the script if it goes crazy
pyautogui.FAILSAFE = True

def start_auto_driver():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(1)
    
    # Set defined resolution so we know the center
    W, H = 640, 480
    cap.set(3, W)
    cap.set(4, H)
    
    # Center of the screen (Where your car is aiming)
    center_x = W // 2

    print("AI Driver Initialized. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, stream=True, classes=[2], conf=0.5, verbose=False)
        
        # Default state: Release keys if nothing is seen
        action = "Idling"
        
        # We need to find the LARGEST car (the one closest to us)
        closest_car = None
        max_area = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                # Filter for the biggest car
                if area > max_area:
                    max_area = area
                    closest_car = (x1, y1, x2, y2)

        # CONTROL LOGIC
        if closest_car:
            x1, y1, x2, y2 = closest_car
            car_center_x = x1 + (x2 - x1) // 2
            
            # Draw box for visual feedback
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame, (center_x, 0), (center_x, H), (255, 255, 255), 1) # Center Line

            # 1. STEERING (Aim at the car)
            # If target is more than 50 pixels to the left
            if car_center_x < (center_x - 50):
                pyautogui.keyDown('a')
                pyautogui.keyUp('d')
                action = "Turning LEFT"
            # If target is more than 50 pixels to the right
            elif car_center_x > (center_x + 50):
                pyautogui.keyDown('d')
                pyautogui.keyUp('a')
                action = "Turning RIGHT"
            else:
                # Target is centered -> Go Straight
                pyautogui.keyUp('a')
                pyautogui.keyUp('d')
                action = "Centered"

            # 2. SPEED (Distance Control)
            # If box width is small (< 150px), we are far away -> GAS
            box_width = x2 - x1
            if box_width < 150:
                pyautogui.keyDown('w')
                pyautogui.keyUp('s')
                action += " + ACCELERATING"
            # If box width is huge (> 300px), we are too close -> BRAKE
            elif box_width > 300:
                pyautogui.keyDown('s')
                pyautogui.keyUp('w')
                action += " + BRAKING"
            else:
                pyautogui.keyUp('w')
                pyautogui.keyUp('s')
        
        else:
            # No car seen? Stop everything safety mechanism
            pyautogui.keyUp('w')
            pyautogui.keyUp('s')
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')

        # Display status
        cv2.putText(frame, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('AI Driver View', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_auto_driver()