import cv2
from cvzone.PlotModule import LivePlot
import mediapipe as mp
import math
import time
import numpy as np
import csv
import os
import pandas as pd

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

csv_file_name = "ear_data_round3.csv"

#Find Last ID
def find_last_person_id():
    data = pd.read_csv(csv_file_name)

    # ดึง person_id จากแถวสุดท้าย
    last_person_id = data['person_id'].iloc[-1] 

    print(f"Last person_id: {last_person_id}")
    return last_person_id

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye):
    A = math.dist(eye[1], eye[5])  # Vertical distance between two points
    B = math.dist(eye[2], eye[4])  # Vertical distance between two points
    C = math.dist(eye[0], eye[3])  # Horizontal distance between two points
    ear = (abs(A) + abs(B)) / (2.0 * abs(C))
    return ear

# Start capturing video
video_path = "assets/Round/video3.mp4"
video_path_almond = "assets/Almond/video1.mp4"

cap = cv2.VideoCapture(0)

#Plot Graph
plotY = LivePlot(640, 360, [0, 5], invert=True)  # Left
plotYRigth = LivePlot(640, 360, [0, 5], invert=True)  # Right


status = ""
left_ear = 0
right_ear = 0

# Check if the CSV file exists
file_exists = os.path.exists(csv_file_name)

with open(csv_file_name, "a", newline="") as csvfile:
    fieldnames = ['person_id', 'eye_type', 'ear_value_left','ear_value_right']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    print(file_exists)
    if not file_exists:
        writer.writeheader()  
        person_id = 1
    else : person_id = find_last_person_id() + 1
    

    # Initialize face mesh detector
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
                    
                    print(left_eye)
                    print(right_eye)
                    
                    # Calculate EAR for both eyes
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)

                    # Determine if the eyes are closed or open
                    if left_ear < 0.25 and right_ear < 0.25:
                        status = "Almond Eyes Open"
                    elif left_ear < 0.33 and right_ear < 0.33:
                        status = "Normal Eyes Open"
                    elif left_ear > 0.35 and right_ear > 0.35:
                        status = "Round Eyes Open"

                    # Draw face landmarks
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                    # Display the EAR and status on the frame
                    cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(frame, f"Status: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                #Write data to csv file
                if left_ear != 0 and right_ear != 0:
                    writer.writerow({
                        'person_id': person_id,
                        'eye_type': 'Almond' if left_ear < 0.25 and right_ear < 0.25 else 'Round',  # กำหนดประเภทตามค่า EAR
                        #Todo : :.3f format decimal####
                        'ear_value_left': f"{left_ear:.3f}",
                        'ear_value_right': f"{right_ear:.3f}"
                    })
                    
            frame = cv2.resize(frame, (540, 700))
            
            imgPlot = plotY.update(left_ear*10)
            cv2.imshow("Plot Rigth eye",imgPlot) 
            
            imgPlotRigth = plotYRigth.update(right_ear*10)
            cv2.imshow("Plot Left eye",imgPlotRigth) 

            # Show the frame
            cv2.imshow('Eye Detection', frame)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


cap.release()
cv2.destroyAllWindows()
