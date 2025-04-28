import joblib
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from cvzone.PlotModule import LivePlot
import math
import time
import csv
import os
import sys
sys.path.append('Stat/type_of_eyes')
import delta


path_file = "ear_data_for_cal_median_ta1.csv"
file_exists = os.path.exists(path_file)

type_eyes = ""
left_ear = 0
right_ear = 0



with open(path_file, "a", newline="") as csvfile:
    fieldnames = ['ear_value_left','ear_value_right']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if not file_exists:
        writer.writeheader()
        

def write_to_csv(datas) :
    with open(path_file, "a", newline="") as csvfile:
        fieldnames = ['ear_value_left','ear_value_right']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        print(file_exists)
        
        for data in datas :    
            writer.writerow({
                'ear_value_left': f"{data[0]}",
                'ear_value_right': f"{data[1]}",
            })

def read_csv_file():
    if file_exists :
        data = pd.read_csv(path_file)
    return data

def validate_type_of_eyes():
    data = read_csv_file()
    ear_value_left = pd.to_numeric(data['ear_value_left'], errors='coerce')
    ear_value_right = pd.to_numeric(data['ear_value_right'], errors='coerce')
    delta_ear_left,delta_ear_right = delta.cal_delta(ear_value_left,ear_value_right)
    type_of_eyes = delta.validate_type_eyes(delta_ear_left,delta_ear_right)
    return type_of_eyes


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
model_round = joblib.load('C:/Project/End/Code/Test/round/model_round2.pkl')
model_almond = joblib.load('C:/Project/End/Code/Test/almond/model_almond2.pkl')

# เปิดกล้อง
file_path_almond = 'C:/Project/End/Code/assets/Almond/video1.mp4'
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    
        check_type_of_eyes = []
    
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
                    
            check_type_of_eyes.append([f"{left_ear:.3f}", f"{right_ear:.3f}"])
            
            if len(check_type_of_eyes) == 502:
                write_to_csv(check_type_of_eyes)
                type_eyes = validate_type_of_eyes()
                time.sleep(2)
                print(type_eyes)
                break
            
            
            
            # #Use Model    
            # prediction = model_round.predict([[left_ear, right_ear]])

            # # แสดงผลการทำนาย
            # if prediction == 0:
            #     cv2.putText(frame, "Eyes Open", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # else:
            #     cv2.putText(frame, "Eyes Closed", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # แสดงภาพ
            cv2.imshow('Driver Drowsiness Detection', frame)

            # ออกจากลูปเมื่อกด 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
