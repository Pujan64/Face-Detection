#pip install opencv-python
#haarcascade_frontalface_default.xml

import cv2

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the video capture object, here 0 means the default camera (usually the webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the captured frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces detected
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame with rectangles drawn around faces
    cv2.imshow('Video', frame)

    # Check for user input to stop the video capturing and face detection loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit if 'q' is pressed
        break
    elif key == ord('s'):  # Stop capturing frames if 's' is pressed
        while True:
            stop_key = cv2.waitKey(1) & 0xFF
            cv2.imshow('Video', frame)  # Show the last frame
            if stop_key == ord('s'):
                break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()


