import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import mediapipe as mp
from collections import deque

# Initialize sound mixer and load alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades for face and eye detection
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_righteye_2splits.xml')

# Load pre-trained Keras model for eye state classification
labels = ['Close', 'Open']
model = load_model('cnnfinal.h5')

# Initialize video capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Initialize variables
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
path = os.getcwd()

# Initialize MediaPipe face mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

start_time = None
alert_playing = False

# Constants for blink detection and gaze direction
BLINK_THRESHOLD = 5
left_eye_landmarks = [468, 469, 470, 471, 472]
right_eye_landmarks = [473, 474, 475, 476, 477]
left_eye_corner = [33, 133]
right_eye_corner = [362, 263]

# Initialize deque for temporal analysis
eye_state_history = deque(maxlen=10)

# Head pose estimation constants
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

def calculate_iris_center(landmarks, indexes, width, height):
    x = sum([landmarks[i].x for i in indexes]) * width / len(indexes)
    y = sum([landmarks[i].y for i in indexes]) * height / len(indexes)
    return x, y

def detect_gaze_direction(landmarks, width, height):
    left_iris_center = calculate_iris_center(landmarks, left_eye_landmarks, width, height)
    right_iris_center = calculate_iris_center(landmarks, right_eye_landmarks, width, height)
    
    left_eye_avg_x = (landmarks[left_eye_corner[0]].x + landmarks[left_eye_corner[1]].x) / 2 * width
    right_eye_avg_x = (landmarks[right_eye_corner[0]].x + landmarks[right_eye_corner[1]].x) / 2 * width

    left_direction = "Left" if left_iris_center[0] < left_eye_avg_x - 5 else "Right" if left_iris_center[0] > left_eye_avg_x + 5 else "Center"
    right_direction = "Left" if right_iris_center[0] < right_eye_avg_x - 5 else "Right" if right_iris_center[0] > right_eye_avg_x + 5 else "Center"

    if left_direction == "Left" and right_direction == "Left":
        return "Looking Left"
    if left_direction == "Right" and right_direction == "Right":
        return "Looking Right"
    return "Looking Center"

def draw_iris(frame, landmarks, indexes, color):
    for i in indexes:
        x = int(landmarks[i].x * frame.shape[1])
        y = int(landmarks[i].y * frame.shape[0])
        cv2.circle(frame, (x, y), 1, color, -1)

def log_drowsiness(log_file, message):
    with open(log_file, 'a') as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def draw_face_rectangle(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

def save_frame(frame, path, filename='image.jpg'):
    cv2.imwrite(os.path.join(path, filename), frame)

def play_alert_sound(sound, alert_playing):
    if not alert_playing:
        sound.play(-1)  # Play sound in a loop
        return True
    return alert_playing

def get_head_pose(shape):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),     # Nose tip
        (shape.part(8).x, shape.part(8).y),       # Chin
        (shape.part(36).x, shape.part(36).y),     # Left eye left corner
        (shape.part(45).x, shape.part(45).y),     # Right eye right corner
        (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)      # Right mouth corner
    ], dtype="double")

    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector, translation_vector

def is_drowsy(eye_state_history):
    if len(eye_state_history) == 0:
        return False
    closed_count = sum(1 for state in eye_state_history if state == 'Close')
    return closed_count / len(eye_state_history) > 0.6

log_file = os.path.join(path, 'drowsiness_log.txt')

SHORT_DROWSINESS_THRESHOLD = 2
PROLONGED_DROWSINESS_THRESHOLD = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    height, width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    draw_face_rectangle(frame, faces)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        predict_x = model.predict(r_eye) 
        rpred = np.argmax(predict_x, axis=1)
        lbl = labels[rpred[0]]
        eye_state_history.append(lbl)
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        predict_x = model.predict(l_eye) 
        lpred = np.argmax(predict_x, axis=1)
        lbl = labels[lpred[0]]
        eye_state_history.append(lbl)
        break

    # Ensure the text area is always visible
    cv2.rectangle(frame, (0, height - 50), (width, height), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(frame, 'Score:' + str(score), (300, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if len(eye_state_history) >= eye_state_history.maxlen and is_drowsy(eye_state_history):
        if start_time is None:
            start_time = time.time()
        elapsed_time = time.time() - start_time
        if elapsed_time > PROLONGED_DROWSINESS_THRESHOLD:
            alert_playing = play_alert_sound(sound, alert_playing)
            cv2.putText(frame, "Eyes Closed: Sleepy!", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            log_drowsiness(log_file, "Eyes Closed: Sleepy!")
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 4)  # Thicker red overlay for prolonged drowsiness
        elif elapsed_time > SHORT_DROWSINESS_THRESHOLD:
            cv2.putText(frame, "Eyes Closed: Drowsy!", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            log_drowsiness(log_file, "Eyes Closed: Drowsy!")
            cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 255), 4)  # Thicker yellow overlay for short drowsiness
        cv2.putText(frame, "Eyes Closed", (50, 50), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
    else:
        start_time = None
        if alert_playing:
            sound.stop()
            alert_playing = False
        cv2.putText(frame, "Eyes Open: Not Sleepy", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0   
    if score > 3:
        save_frame(frame, path)
        if thicc < 16:
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc) 

    # Process face landmarks with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw iris
            draw_iris(frame, face_landmarks.landmark, left_eye_landmarks, (0, 0, 255))
            draw_iris(frame, face_landmarks.landmark, right_eye_landmarks, (0, 255, 0))
            
            # Detect gaze direction
            if not is_drowsy(eye_state_history):
                gaze_direction = detect_gaze_direction(face_landmarks.landmark, width, height)
                cv2.putText(frame, gaze_direction, (50, 50), font, 1, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()