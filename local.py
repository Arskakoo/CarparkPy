import cv2
import numpy as np

# Predefined parking spots
detected_parking_spots = [
   (31, 653, True), 
   (126, 663, True),
   (204, 539, True), 
   (244, 684, True),
   (316, 550, True), 
   (374, 679, True), 
   (412, 551, False), 
   (495, 674, True),
   (504, 560, False), 
   (600, 558, True), 
   (623, 682, True),
   (699, 562, False),
   (749, 693, True),
   (802, 563, True),
   (887, 693, True),
   (901, 565, True),
   (1000, 564, True),
   (1028, 700, True),
   (1109, 566, True),
   (1174, 704, True),
   (1255, 575, False),
   (1317, 722, False),
   (1382, 578, False),
   (1539, 727, False)
]


# Function to process each frame
def process_frame(frame):
    # Resize the image
    width = 1600
    height = 900
    resized_image = cv2.resize(frame, (width, height))

    # Draw static dots for detected parking spots
    for center_x, center_y, is_taken in detected_parking_spots:
        color = (0, 0, 255) if is_taken else (0, 255, 0)  # Red for taken, green for free
        cv2.circle(resized_image, (center_x, center_y), 10, color, -1)  # Draw static dot

    return resized_image

# Main function to run the detection
def main(video_source="image/Example.jpg"):  # Change to a video file path if needed
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame)

        # Show the processed frame
        cv2.imshow('Predefined Parking Spots', processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()  # You can pass a video filename to process a video file
