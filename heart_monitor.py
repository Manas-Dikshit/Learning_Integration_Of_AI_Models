import cv2
import numpy as np
import time
import json
from scipy.signal import butter, lfilter

face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

signal = []
timestamps = []
bpm_values = []

start_time = time.time()

def bandpass(data, low, high, fs):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(2, [low, high], btype="band")
    return lfilter(b, a, data)

def moving_average(data, window=5):
    if len(data) < window:
        return np.mean(data)
    return np.mean(data[-window:])

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Use forehead region instead of entire face
        fh_y = int(y + h * 0.15)
        fh_h = int(h * 0.15)

        roi = frame[fh_y:fh_y+fh_h, x:x+w]

        green = np.mean(roi[:, :, 1])

        signal.append(green)
        timestamps.append(time.time())

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        if len(signal) > 150:

            fps = len(timestamps) / (timestamps[-1] - timestamps[0])

            filtered = bandpass(np.array(signal), 0.8, 3, fps)

            fft = np.fft.rfft(filtered)
            freqs = np.fft.rfftfreq(len(filtered), 1/fps)

            bpm = freqs[np.argmax(np.abs(fft))] * 60

            # Filter unrealistic values
            if 40 < bpm < 180:
                bpm_values.append(bpm)

            # Smooth BPM
            smooth_bpm = moving_average(bpm_values)

            cv2.putText(frame,
                        f"HR: {int(smooth_bpm)} BPM",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

            signal.pop(0)
            timestamps.pop(0)

    cv2.imshow("Video HR Monitor", frame)

    # After 1 minute save results
    if time.time() - start_time >= 60:

        if len(bpm_values) > 0:
            avg_bpm = sum(bpm_values) / len(bpm_values)
        else:
            avg_bpm = 0

        result = {
            "duration_seconds": 60,
            "readings": bpm_values,
            "average_bpm": avg_bpm,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open("heart_rate_results.json", "w") as f:
            json.dump(result, f, indent=4)

        print("Final Average BPM:", avg_bpm)
        print("Results saved to heart_rate_results.json")

        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
