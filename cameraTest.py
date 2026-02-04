import cv2

def start_processing_stream():
    cap = cv2.VideoCapture(1)

    # LOWER THE RESOLUTION
    # This is crucial for real-time processing (FPS). 
    # High-res images choke the CPU/AI model unnecessary.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # STEP 1: Convert to Grayscale
        # Color data is computationally expensive (3 channels). 
        # Structure/Edge detection only needs light intensity (1 channel).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # STEP 2: Gaussian Blur
        # This smooths the image slightly to reduce "noise" so we don't 
        # detect dust or texture as an edge.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # STEP 3: Canny Edge Detection
        # The arguments (50, 150) are the thresholds for what counts as an edge.
        # You might need to tune these for your specific game screen brightness.
        edges = cv2.Canny(blurred, 50, 150)

        # Display both views
        cv2.imshow('Raw Input', frame)
        cv2.imshow('AI Vision (Edges)', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_processing_stream()