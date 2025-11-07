import cv2

# Start webcam
video_capture = cv2.VideoCapture(0)

# Load face cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Live Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
video_capture.release()
cv2.destroyAllWindows()
