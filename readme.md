
# Driver Drowsiness Detection System

This repository contains a Python-based implementation of a **Driver Drowsiness Detection System** using **OpenCV, Dlib, Pygame**, and **Streamlit**. The system uses real-time webcam video feed to monitor the driver's eye aspect ratio (EAR) and triggers an alert when drowsiness is detected.

This project is intended for **academic and educational purposes**, showcasing how computer vision and facial landmark detection can be used for driver safety applications.

---

## Overview

The drowsiness detection system monitors eye closure using facial landmarks. When a driver's eyes remain closed beyond a threshold time, an **alarm sound** is triggered to alert them. The system supports two interfaces for flexibility and usability:

### Supported Interfaces:

| Version       | Interface | Use Case                          |
|---------------|-----------|-----------------------------------|
| OpenCV Script | Terminal  | Lightweight, good for testing     |
| Streamlit App | Browser   | User-friendly, demo/presentation  |

Both interfaces use the same underlying detection logic based on facial landmarks and EAR (Eye Aspect Ratio) computation.

---

## Features

- **Real-Time Detection**: Uses the webcam to detect eye movement and drowsiness in real-time.
- **EAR-based Detection**: Computes Eye Aspect Ratio using Dlib’s 68 facial landmarks.
- **Alert System**: Plays a sound alarm (`music.wav`) when drowsiness is detected.
- **Modular Design**: Shared logic can be reused across OpenCV CLI and Streamlit GUI.
- **Streamlit GUI**: A clean interface to run and display detection results in the browser.
- **OpenCV Script**: A terminal version that is fast, lightweight, and test-friendly.

---

## System Architecture

The detection process includes the following stages:

1. **Face Detection**: Uses Dlib’s frontal face detector.
2. **Landmark Detection**: Uses `shape_predictor_68_face_landmarks.dat` for facial landmark estimation.
3. **Eye Aspect Ratio Calculation**: Uses specific landmarks around the eyes to compute EAR.
4. **Drowsiness Check**: Compares EAR with a threshold and keeps count of consecutive low-EAR frames.
5. **Alarm Triggering**: If low EAR persists, triggers the `music.wav` alarm.
6. **User Interfaces**: 
   - **OpenCV Script**: Visual output via `cv2.imshow()`
   - **Streamlit App**: GUI with a button to start detection

---

## Simulation / Running

### OpenCV CLI Version

Run the script via terminal:

```bash
python drowsiness_detector.py
```

This version opens a webcam window, displays eye contours, and prints alert flags in terminal. Press `q` to exit.

---

### Streamlit GUI Version

Run in browser using:

```bash
streamlit run app.py
```

This will open a Streamlit app in your default browser with a button to begin detection.

---

## Setup & Dependencies

Install all dependencies using:

```bash
pip install streamlit opencv-python imutils numpy scipy dlib pygame
```

Make sure you have the following files:

- `models/shape_predictor_68_face_landmarks.dat` – Dlib model (~100MB+)
- `music.wav` – Alarm sound
- `drowsiness_detector.py` – Core logic (OpenCV)
- `app.py` – Streamlit GUI wrapper

---

## Folder Structure

```
DriverDrowsinessDetection/
│
├── app.py                             # Streamlit GUI wrapper
├── drowsiness_detector.py            # Core OpenCV logic
├── music.wav                         # Alarm sound file
├── models/
│   └── shape_predictor_68_face_landmarks.dat  # Facial landmark model
├── README.md
```

---

## Modularization Strategy

Both interfaces (CLI and GUI) can share the same core detection logic by abstracting detection into a function/module:

```python
def run_drowsiness_detection(video_source=0):
    # shared logic
```

This can be imported into both the Streamlit app and the OpenCV script for consistency and reuse.

---

## Acknowledgments

- This project was developed for academic and demonstration purposes.
- Special thanks to the OpenCV, Dlib, and Streamlit communities for providing powerful tools and resources.
- Inspired by real-world driver safety challenges and applications in transportation.

