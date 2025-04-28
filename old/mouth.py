import cv2 as cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time
import timeit

# --- Configuration ---
video_path = "assets/video2.mp4"  # Change to 0 for webcam
rectengle_top_left = (100,50)
rectengle_bottom_rigth = (500,450)
blink_threshold = 32.75 # The threshold that will count as blink.
blink_delay = 1.15  # Minimum time between blinks (in seconds)
list_ratio_length = 10 # Number of list to keep.

# --- Initialize Video Capture ---
cap = cv2.VideoCapture(0)  # Use 0 for webcam
if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# --- Initialize FaceMesh Detector ---
detector = FaceMeshDetector(maxFaces=2)

# --- Initialize Plotting ---
plotY = LivePlot(640, 360, [25, 40], invert=True)  # Left plot
plotMouth = LivePlot(640, 360, [0, 40], invert=True)  # Left plot

# --- Landmark IDs ---
idListRigth = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130]
idListLeft = [362, 381, 380, 374, 373, 390, 263, 388, 386, 384, 385, 387]
mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,270,269,267,0,37,39,40,185,13,14]

# --- Global Variables ---
listRatioRigth = []  # List to store recent eye aspect ratios
listRatioLeft = []
listRatioMouth = []
blinkCount = 0  # Counter for detected blinks
last_blink_time = 0  # Time of the last detected blink
face_detected = False # Detect if the face is in the frame

# --- Main Loop ---
while True:
    # --- Read Frame ---
    ret, frame = cap.read()  # Read the full-size frame
    if not ret:
        print("End of video or error reading frame.")
        break

    # --- Find Face Mesh ---
    frame, faces = detector.findFaceMesh(frame, draw=False)

    # --- Process Detected Faces ---
    if faces:
        face_detected = True
        face = faces[0]  # Get the first face detected

        # --- Find Bounding Box of Face Mesh ---
        min_x = min(face[i][0] for i in range(len(face)))
        max_x = max(face[i][0] for i in range(len(face)))
        min_y = min(face[i][1] for i in range(len(face)))
        max_y = max(face[i][1] for i in range(len(face)))

        # --- Draw Face Bounding Box ----
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue box

        # --- Convert Landmarks to Full Frame Coordinates ---
        # Convert coordinates back to the original frame size
        valid_landmarks = True
        if (rectengle_top_left[0] <= min_x and max_x <= rectengle_bottom_rigth[0]) and (
            rectengle_top_left[1] <= min_y and max_y <= rectengle_bottom_rigth[1]
        ):
            valid_landmarks = True
        else : valid_landmarks = False



        if valid_landmarks:
            # --- Draw Landmarks ---
            for id in idListRigth:
                cv2.circle(frame, face[id], 2, (255, 0, 255), cv2.FILLED)

            for id in idListLeft:
                cv2.circle(frame, face[id], 2, (255, 0, 255), cv2.FILLED)
            
            for id in mouth:
                cv2.circle(frame, face[id], 2, (255, 0, 255), cv2.FILLED)

            try:
                mouthTop = face[13]
                mouthBottom  = face[14]
                mouthLeft = face[61]
                mouthRigth = face[291]
            except IndexError:
                continue

            # --- Calculate Length of Mouth ---
            lengthHorMouth, _ = detector.findDistance(mouthTop, mouthBottom)
            lengthVerMouth, _ = detector.findDistance(mouthLeft, mouthRigth)
            ratioMouth = int((lengthHorMouth / lengthVerMouth) * 100)
            
            # --- Process Eye Aspect Ratio (Right) ---
            listRatioMouth.append(ratioMouth)
            if len(listRatioMouth) >= list_ratio_length:
                listRatioMouth.pop(0)
            ratioAvgMouth = sum(listRatioMouth) / len(listRatioMouth)
        else:
            print("Face outside the green box")
        
        # --- Display Text ---
        # cvzone.putTextRect(frame, f"Blink Count : {blinkCount}", (100, 100))
        cvzone.putTextRect(frame, f"Mouth length : {ratioAvgMouth}", (100, 100))
            # --- Draw Rectangle ---
        cv2.rectangle(frame,rectengle_bottom_rigth,rectengle_top_left, (0, 255, 0), 2)  # Draw on frame
    
        # --- Display Frame ---
        cv2.imshow('Eye Detection', frame)  # Show frame (with rectangle and landmarks)
        # Todo : Plot Left
        imgPlotMouth = plotMouth.update(ratioAvgMouth)
        cv2.imshow("Plot Mouth",imgPlotMouth)
        # imgPlotRight = plotY.update((ratioAvgRight + ratioAvgLeft)/2)
        # cv2.imshow("Plot Rigth eye",imgPlotRight)
    else:
        face_detected = False
        cvzone.putTextRect(frame, f"No Face Detected", (100, 100))
        cv2.imshow(f'Eye Detection', frame)
    # --- Exit on 'q' ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
