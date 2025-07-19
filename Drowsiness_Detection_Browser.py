import streamlit as st
import cv2
import imutils
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer

# EAR calculation function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Streamlit app title
st.title("Driver Drowsiness Detection")
st.markdown("Click the button below to start drowsiness detection.")

# Load Dlib models
predictor_path = "/Users/chaudharysajjadhussain/Downloads/MSc_Project/Driver Drowsiness Detection/models/shape_predictor_68_face_landmarks.dat"
music_path = "/Users/chaudharysajjadhussain/Downloads/MSc_Project/Driver Drowsiness Detection/music.wav"

try:
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor(predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
except RuntimeError as e:
    st.error("Failed to load Dlib models. Make sure the .dat file path is correct.")
    st.stop()

# Initialize pygame mixer for sound
try:
    mixer.init()
    mixer.music.load(music_path)
except FileNotFoundError:
    st.error("Failed to find the 'music.wav' file. Make sure the file path is correct.")
    st.stop()

# Button to start the camera detection
start_detection = st.button("Start Camera")

if start_detection:
    cap = cv2.VideoCapture(0)
    flag = 0
    thresh = 0.25
    frame_check = 20

    # Temporary placeholder for webcam feed
    stframe = st.empty()
    stop_button = st.button("Stop Camera")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            # Alert if EAR is below the threshold
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Play alarm sound
                    mixer.music.play()
            else:
                flag = 0

            # Draw contours around the eyes
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Display the webcam feed in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
    st.success("Camera stopped.")

