from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import csv
import time

mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks (2).dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
left_flag = 0s
right_flag = 0

# Data Logging
log_file = 'drowsiness_log.csv'  

# Open the log file in append mode 
with open(log_file, 'a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(['Timestamp', 'Eye Aspect Ratio'])

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if leftEAR < thresh:
            left_flag += 1
            if left_flag >= frame_check:
                cv2.putText(frame, "ALERT! Drowsiness Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()

                # Log the drowsiness event with timestamp and eye aspect ratio
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                with open(log_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, leftEAR])

        else:
            left_flag = 0

        if rightEAR < thresh:
            right_flag += 1
            if right_flag >= frame_check:
                cv2.putText(frame, "ALERT! Drowsiness Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()

                # Log the drowsiness event with timestamp and eye aspect ratio
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                with open(log_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, rightEAR])

        else:
            right_flag = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
timestamps = []
ear_values = []

with open(log_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read and store the header row

    for row in reader:
        try:
            timestamp = row[0]
            ear_value = float(row[1])
            timestamps.append(timestamp)
            ear_values.append(ear_value)
        except ValueError:
            continue  # Skip rows that cannot be converted to a float

plt.plot(timestamps, ear_values)
plt.xlabel('Timestamp')
plt.ylabel('Eye Aspect Ratio')
plt.title('Drowsiness Events Over Time')
plt.xticks(rotation=45)
plt.show()
