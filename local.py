import cv2
import numpy as np

# Predefined parking spots
detected_parking_spots = [
    # x, y, is_taken
    (100, 685, True),  # vasemmalta oikealle 1
    (500, 685, True),  # oikealta vasemmalle 1
    (620, 685, True),  # oikealta vasemmalle 6
    (750, 685, True),  # oikealta vasemmalle 5
    (895, 685, True),  # oikealta vasemmalle 4
    (1050, 685, True),  # oikealta vasemmalle 3
    (1190, 725, True),  # oikealta vasemmaalle 2
    (1355, 730, False),  # oikealta vasemmalle 1
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
