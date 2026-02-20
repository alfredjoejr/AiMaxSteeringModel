import cv2
import requests
import numpy as np

# PASTE THE NGROK URL FROM YOUR COLAB OUTPUT HERE:
COLAB_URL = "https://9087-34-125-166-168.ngrok-free.app/detect" 

# Connect to the external camera
cap = cv2.VideoCapture(1) 

print("Streaming video to Colab GPU... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # Compress the resolution slightly to prevent massive lag over the internet
    frame = cv2.resize(frame, (640, 480))

    # Encode frame to JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)

    try:
        # HTTP POST the frame to Colab
        response = requests.post(
            COLAB_URL, 
            data=img_encoded.tobytes(), 
            headers={'Content-Type': 'application/octet-stream'}
        )
        
        if response.status_code == 200:
            # Decode and display the image returned by Colab (now with bounding boxes)
            nparr = np.frombuffer(response.content, np.uint8)
            result_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            cv2.imshow('Remote AI Vision', result_frame)
        else:
            print(f"Server error: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Network error (Is Colab running?): {e}")

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()