import os
print("Current working directory:", os.getcwd())
import os
os.chdir("/home/hammad/Downloads/AI course/DROWSINESS DETECTION/Driver Drowsiness Detection-20250505T151232Z-1-001/Driver Drowsiness Detection")

from pygame import mixer
mixer.init()
mixer.music.load("music.wav")

import cv2
import imutils
from pygame import mixer
from scipy.spatial import distance
import dlib
from imutils import face_utils

# Initialize the mixer for playing sound
mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold and frame check parameters
thresh = 0.25
frame_check = 20

# Load face detector and landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Define the indices for the eyes landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize the webcam
cap = cv2.VideoCapture(0)

flag = 0
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        # Extract the eyes landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calculate Eye Aspect Ratio (EAR)
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Draw contours around the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Trigger the alert if EAR is below the threshold
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0
    
    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit the loop on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
subjects = detect(gray, 0)
print("Faces detected:", len(subjects))

