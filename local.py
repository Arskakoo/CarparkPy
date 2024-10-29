import cv2
import numpy as np
import time
from datetime import datetime
import os

# Initialize webcam and background subtractor
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Predefined parking spots (coordinates with initial False for no movement detected)
detected_parking_spots = [
    (887, 693, False), (1028, 700, False), (1174, 704, False),
    (1317, 722, False), (749, 693, False), (623, 682, False),
    (495, 674, False), (374, 679, False), (244, 684, False),
    (126, 663, False), (31, 653, False), (204, 539, False),
    (316, 550, False), (504, 560, False), (412, 551, False),
    (600, 558, False), (699, 562, False), (802, 563, False),
    (901, 565, False), (1000, 564, False), (1109, 566, False),
    (1255, 575, False), (1382, 578, False), (1539, 727, False),
]

# Parameters
radius = 20
motion_threshold = 50  # Tune this for sensitivity to motion
motion_log_file = "motion_log.txt"

# Ensure the log file exists
if not os.path.exists(motion_log_file):
    with open(motion_log_file, "w") as f:
        f.write("Timestamp,Parking Spot,Occupied\n")

def detect_motion(roi, threshold):
    """Detect motion in the given region of interest (ROI)."""
    motion = np.count_nonzero(roi)
    return motion > threshold

def draw_parking_spots(frame, spots, radius):
    """Draw circles around each parking spot, indicating motion detection."""
    for i, (x, y, movement_detected) in enumerate(spots):
        color = (0, 0, 255) if movement_detected else (0, 255, 0)  # Red if motion, else green
        cv2.circle(frame, (x, y), radius, color, 2)
        # Display parking spot status
        status = "Taken" if movement_detected else "Free"
        cv2.putText(frame, f"Spot {i+1}: {status}", (x - 40, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Start time for FPS calculation
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Resize for better performance (optional)
    frame = cv2.resize(frame, (1600, 900))
    
    # Apply background subtraction to detect motion
    fgmask = fgbg.apply(frame)

    # Detect motion in each parking spot
    for i, (x, y, _) in enumerate(detected_parking_spots):
        # Extract ROI around the parking spot
        roi = fgmask[max(y - radius, 0):min(y + radius, fgmask.shape[0]),
                     max(x - radius, 0):min(x + radius, fgmask.shape[1])]
        
        if roi.shape[0] != 2 * radius or roi.shape[1] != 2 * radius:
            print(f"Skipping ROI at ({x},{y}) due to incorrect size.")
            continue
        
        # Update motion detection status
        movement_detected = detect_motion(roi, motion_threshold)
        detected_parking_spots[i] = (x, y, movement_detected)

        # Log motion events
        if movement_detected:
            with open(motion_log_file, "a") as f:
                f.write(f"{datetime.now()},Spot {i + 1},Occupied\n")

    # Draw circles and status on frame
    draw_parking_spots(frame, detected_parking_spots, radius)

    # Display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Get current time
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {current_time}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the output frame
    cv2.imshow("Parking Spot Monitoring", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
