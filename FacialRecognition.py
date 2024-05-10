import cv2 as cv
import threading
import face_recognition
import subprocess
from twilio.rest import Client
import numpy as np

# Twilio credentials
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
twilio_phone_number = 'your_twilio_phone_number'
# The phone number you want to send the message to
recipient_phone_number = 'your phone number'

client = Client(account_sid, auth_token)


def send_sms(message):
    try:
        message = client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=recipient_phone_number
        )
        print(f"Message sent successfully. SID: {message.sid}")
    except Exception as e:
        print(f"Error sending message: {str(e)}")


cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_matches = {}

reference_images = {
    'Sid Reference1': '/Users/siddarthakalankam/Downloads/Reference.PNG',
    'Sid Reference2': '/Users/siddarthakalankam/Downloads/Reference2.jpg',
    'Sid Reference3': '/Users/siddarthakalankam/Downloads/Reference3.PNG',
    'Dad Reference1': '/Users/siddarthakalankam/Downloads/Dad_Reference1.jpg',
    'Dad Reference2': '/Users/siddarthakalankam/Downloads/Dad_Reference2.jpg',
    'Dad Reference3': '/Users/siddarthakalankam/Downloads/Dad_Reference3.jpg',
    'Mom Reference1': '/Users/siddarthakalankam/Downloads/Mom_Reference1.jpg',
    'Mom Reference2': '/Users/siddarthakalankam/Downloads/Mom_Reference2.jpg',
    'Mom Reference3': '/Users/siddarthakalankam/Downloads/Mom_Reference3.jpg'
}

reference_encodings = {}
notification_sent = False

# Load reference images and generate encodings
for name, image_path in reference_images.items():
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    reference_encodings[name] = encoding

# Haar cascade for face detection
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


def send_notifications(message):
    send_sms(message)


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:  # Execute every 30 frames
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                # Extract the detected face region
                face_image = frame[y:y+h, x:x+w]
                # Convert to RGB format
                face_image = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
                frame_encoding = face_recognition.face_encodings(face_image)

                if len(frame_encoding) > 0:
                    matches = face_recognition.compare_faces(
                        list(reference_encodings.values()), frame_encoding[0], tolerance=0.5)
                    face_matches = {name: match for name, match in zip(
                        reference_encodings.keys(), matches)}
                else:
                    face_matches = {}
            else:
                face_matches = {}

        counter += 1

        if len(faces) > 0:
            match_found = False  # Flag to track if any match is found
            unknown_person_detected = False  # Flag to track if an unknown person is detected

            for (x, y, w, h) in faces:
                # Draw rectangle around the detected face
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            for name, match in face_matches.items():
                if match:
                    cv.putText(frame, 'MATCH', (100, 100),
                               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    match_found = True
                    break

            if not match_found:
                cv.putText(frame, 'NO MATCH: Unknown Person', (20, 450),
                           cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                if not unknown_person_detected and not notification_sent:
                    send_notifications('Unknown person detected!')
                    notification_sent = True
                    unknown_person_detected = True

        cv.imshow("WebCam Video", frame)
        key = cv.waitKey(1)

        if key == ord('q'):
            break

cv.destroyAllWindows()
