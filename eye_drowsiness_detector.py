import cv2
import numpy as np
import mediapipe as mp
import time
import pygame

# -----------------------------
# Alarm setup
# -----------------------------
pygame.mixer.init()
pygame.mixer.music.load("siren.mp3")

# -----------------------------
# MediaPipe FaceLandmarker setup
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="eye_state/face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

face_landmarker = FaceLandmarker.create_from_options(options)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# -----------------------------
# Parameters
# -----------------------------
EAR_THRESHOLD = 0.25
CLOSED_TIME_LIMIT = 5

closed_start = None
alarm_played = False


def eye_aspect_ratio(eye):

    A = np.linalg.norm(eye[1]-eye[5])
    B = np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])

    return (A+B)/(2.0*C)


while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

    results = face_landmarker.detect_for_video(
        mp_image,
        int(time.time()*1000)
    )

    if results.face_landmarks:

        landmarks = results.face_landmarks[0]

        h,w,_ = frame.shape

        left_eye_idx = [33,160,158,133,153,144]
        right_eye_idx = [362,385,387,263,373,380]

        left_eye = np.array(
            [(landmarks[i].x*w,landmarks[i].y*h) for i in left_eye_idx]
        )

        right_eye = np.array(
            [(landmarks[i].x*w,landmarks[i].y*h) for i in right_eye_idx]
        )

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)

        ear = (ear_left+ear_right)/2

        if ear < EAR_THRESHOLD:

            if closed_start is None:
                closed_start = time.time()

            elapsed = time.time() - closed_start

            cv2.putText(frame,"Eyes Closed",(30,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            if elapsed > CLOSED_TIME_LIMIT and not alarm_played:
                print("ALERT! Eyes closed too long")
                pygame.mixer.music.play()
                alarm_played = True

        else:

            closed_start = None
            alarm_played = False

            cv2.putText(frame,"Eyes Open",(30,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Eye Drowsiness Detector",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()