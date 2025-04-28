import joblib
import numpy as np
import cv2
import mediapipe as mp
from cvzone.PlotModule import LivePlot
import math

def calculate_ear(eye):
    A = math.dist(eye[1], eye[5])  # Vertical distance between two points
    B = math.dist(eye[2], eye[4])  # Vertical distance between two points
    C = math.dist(eye[0], eye[3])  # Horizontal distance between two points
    ear = (abs(A) + abs(B)) / (2.0 * abs(C))
    return ear

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# โหลดโมเดลที่บันทึกไว้ C:/Project/End/Code/ear_data.csv
model = joblib.load('C:/Project/End/Code/Test/round/model_round2.pkl')

# เปิดกล้อง
file_path_almond = 'C:/Project/End/Code/assets/Almond/video1.mp4'
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            
            #Close Video
            if not ret:
                break

            # Convert the image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find landmarks
            results = face_mesh.process(image)

            # Convert back to BGR for OpenCV
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Define the landmarks for the left and right eyes
                    left_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [33, 160, 158, 133, 153, 144]] #33, 160, 158, 133, 153, 144 || 159,163,161,145,157,154
                    right_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [362, 385, 387, 263, 373, 380]]
                    
                    # Calculate EAR for both eyes
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    
            prediction = model.predict([[left_ear, right_ear]])
            
            print(f"{left_ear} {right_ear} {prediction}\n")

            # แสดงผลการทำนาย
            if prediction == 0:
                cv2.putText(frame, "Eyes Open", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Eyes Closed", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # แสดงภาพ
            cv2.imshow('Driver Drowsiness Detection', frame)

            # ออกจากลูปเมื่อกด 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
