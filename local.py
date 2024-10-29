import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

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

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Radius for parking spot circles
radius = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for better performance (optional)
    frame = cv2.resize(frame, (1600, 900))
    
    # Apply background subtraction to detect motion
    fgmask = fgbg.apply(frame)
    
    # Detect movement within each defined circle
    for i, (x, y, movement_detected) in enumerate(detected_parking_spots):
        # Extract the region of interest (ROI) around the parking spot
        roi = fgmask[y - radius:y + radius, x - radius:x + radius]
        
        # Check for motion within the ROI by counting non-zero pixels
        motion = np.count_nonzero(roi)
        
        # Set threshold for motion detection
        if motion > 50:
            color = (0, 0, 255)  # Red for movement detected
            detected_parking_spots[i] = (x, y, True)
        else:
            color = (0, 255, 0)  # Green for no movement
            detected_parking_spots[i] = (x, y, False)
        
        # Draw the circle on the frame
        cv2.circle(frame, (x, y), radius, color, 2)
    
    # Show the output frame
    cv2.imshow("Parking Spot Monitoring", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
