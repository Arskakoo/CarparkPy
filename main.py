import cv2
import numpy as np
import time  

cap = cv2.VideoCapture(0)

detected_parking_spots = [
    (887, 693, False, 0), (1028, 700, False, 0), (1174, 704, False, 0),
    (1317, 722, False, 0), (749, 693, False, 0), (623, 682, False, 0),
    (495, 674, False, 0), (374, 679, False, 0), (244, 684, False, 0),
    (126, 663, False, 0), (31, 653, False, 0), (204, 539, False, 0),
    (316, 550, False, 0), (504, 560, False, 0), (412, 551, False, 0),
    (600, 558, False, 0), (699, 562, False, 0), (802, 563, False, 0),
    (901, 565, False, 0), (1000, 564, False, 0), (1109, 566, False, 0),
    (1255, 575, False, 0), (1382, 578, False, 0), (1539, 727, False, 0),
]

radius = 20

ret, ref_frame = cap.read()
if not ret:
    print("Failed to capture reference frame.")
    cap.release()
    exit()
    
ref_frame = cv2.resize(ref_frame, (1600, 900))
gray_ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
gray_ref_frame = cv2.GaussianBlur(gray_ref_frame, (5, 5), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1600, 900))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    for i, (x, y, was_movement, last_movement_time) in enumerate(detected_parking_spots):
        ref_roi = gray_ref_frame[max(0, y - radius):y + radius, max(0, x - radius):x + radius]
        current_roi = gray_frame[max(0, y - radius):y + radius, max(0, x - radius):x + radius]
        if ref_roi.shape != current_roi.shape:
            print(f"ROI shape mismatch at spot {i}. Skipping.")
            continue
        diff = cv2.absdiff(ref_roi, current_roi)
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion = np.count_nonzero(diff)
        is_movement = motion > 500  
        current_time = time.time()
        if is_movement and not was_movement:
            print(f"Significant movement detected: Spot ID {i}")
            detected_parking_spots[i] = (x, y, True, current_time)  
        elif not is_movement and was_movement:
            if current_time - last_movement_time >= 5:
                print(f"No significant movement: Spot ID {i}")
                detected_parking_spots[i] = (x, y, False, 0)  
        else:
            if is_movement:
                color = (0, 0, 255)  
            else:
                if current_time - last_movement_time >= 5:
                    color = (0, 255, 0)  
                else:
                    color = (0, 0, 255) 
        detected_parking_spots[i] = (x, y, is_movement, current_time if is_movement else last_movement_time)
        cv2.circle(frame, (x, y), radius, color, 2)
    cv2.imshow("Parking Spot Monitoring", frame)
    
    # Exit q    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
