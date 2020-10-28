import cv2
# import face_recognition

def demo():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '../data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '../data/haarcascade_eye.xml')

    # To capture video from webcam.
    cap = cv2.VideoCapture(0)
    # To use a video file as input
    # cap = cv2.VideoCapture('filename.mp4')

    while True:
        # Read the frame
        ret, img = cap.read()
        # Convert to grayscale
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # faces = face_recognition.face_locations(img)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Draw eyes
                roi_color = img[y:y + h, x:x + w]
                roi_gray = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eyes = eye_cascade.detectMultiScale(gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # TODO: process eyes' detections
            # for (top, right, bottom, left) in faces:
            #     cv2.rectangle(img, (left, bottom), (right, top), (255, 0, 0), 2)
            # Display
            cv2.imshow('FaceMaskDetection', img)
            # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    # Release the VideoCapture object
    cap.release()


if __name__ == "__main__":
    demo()