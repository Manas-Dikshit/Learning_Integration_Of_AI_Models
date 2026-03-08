# eye_monitor.py

import cv2
import time
import threading
from playsound import playsound

ALARM_FILE = "./siren.mp3"
CLOSED_LIMIT = 5   # seconds

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

cap = cv2.VideoCapture(0)

eyes_closed_start = None


def play_alarm():
    playsound(ALARM_FILE)


while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = False

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) > 0:
            eyes_detected = True

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

    if not eyes_detected:

        if eyes_closed_start is None:
            eyes_closed_start = time.time()

        elapsed = time.time() - eyes_closed_start

        if elapsed > CLOSED_LIMIT:
            cv2.putText(frame, "WAKE UP!", (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

            threading.Thread(target=play_alarm).start()

    else:
        eyes_closed_start = None

    cv2.imshow("Eye Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()