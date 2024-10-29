import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

# Predefined parking spots
detected_parking_spots = [
    (100, 685, True),   # vasemmalta oikealle 1
    (500, 685, True),   # oikealta vasemmalle 1
    (620, 685, True),   # oikealta vasemmalle 6
    (750, 685, True),   # oikealta vasemmalle 5
    (895, 685, True),   # oikealta vasemmalle 4
    (1050, 685, True),  # oikealta vasemmalle 3
    (1190, 725, True),  # oikealta vasemmaalle 2
    (1355, 730, False),  # oikealta vasemmalle 1
]

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

def gen_frames():  
    cap = cv2.VideoCapture(0)  # Use 0 for webcam; replace with video source if needed

    while True:
        success, frame = cap.read()  # Read a frame from the video
        if not success:
            break
        
        processed_frame = process_frame(frame)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        # Yield the frame in a format compatible with Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Parking Spot Detection</title>
    </head>
    <body>
        <h1>Parking Spot Detection</h1>
        <img src="/video_feed" width="1600" height="900">
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)  # Run the Flask server
