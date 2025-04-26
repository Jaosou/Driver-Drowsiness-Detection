import cv2 as cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time
import timeit

video_path = "assets/video2.mp4"

# * Web Cam Video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()
detector = FaceMeshDetector(maxFaces=2)

last_blink_time = 0

#* detection zone
frame_width = 700  # Adjust as needed
frame_height = 440  # Adjust as needed

# * Plot Graph and Fix window
plotY = LivePlot(640, 360, [25, 40], invert=True)  # Left
plotYRight = LivePlot(640, 360, [25, 40], invert=True)  # Right

# * Point landmark
idListRigth = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130]
idListRigthRight = [362, 381, 380, 374, 373, 390, 263, 388, 386, 384, 385, 387]
#idMouth = [61, 62, 185, 40, 39, 37, 0, 267, 269, 270, 300]

listRatioRigth = []
listRatioRigthRight = []
blinkCount = 0
counter = 0
face_detected = False # add a flag to check if face is detec

# * Main
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break
    
    frame, faces = detector.findFaceMesh(frame, draw=False)
    
    if faces:
        # face_detected = True # set face detected to true
        face = faces[0]
        # * Mark point on Face
        for id in idListRigth:
            cv2.circle(frame, face[id], 2, (255, 0, 255), cv2.FILLED)
        """ for id in idListRigthRight:
            cv2.circle(frame,face[id],2,(255,0,255),cv2.FILLED)
        for id in idMouth:
            #Todo : (255,0,255) = color
            cv2.circle(frame,face[id],2,(255,0,255),cv2.FILLED) """

        # * Left Eye
        try:
            leftUp = face[159]
            leftDown = face[23]
            leftleft = face[130]
            leftRight = face[243]
        except IndexError:
            continue

        # * Right Eye
        try:
            rightUp = face[386]
            rightDown = face[374]
            rightleft = face[362]
            rightRight = face[263]
        except (IndexError, UnboundLocalError):
            continue

        # * Horizon length Left Calulate Distance between point.
        lengthHor, _ = detector.findDistance(leftUp, leftDown)
        # * Vertical Length Left Calulate Distance between point.
        lengthVer, _ = detector.findDistance(leftleft, leftRight)

        # * Process Left eye
        ratio = int((lengthHor / lengthVer) * 100)
        listRatioRigth.append(ratio)
        if len(listRatioRigth) >= 8:
            listRatioRigth.pop(0)
        ratioAvg = sum(listRatioRigth) / len(listRatioRigth)

        # * Blink Count
        current_time = timeit.default_timer()
        if ratioAvg < 33.5 and (current_time - last_blink_time) > 1.15:
            print("Blink",(current_time - last_blink_time))
            blinkCount += 1
            last_blink_time = timeit.default_timer()

        # if counter != 0:
        #     counter += 1
        #     if counter > 25:
        #         counter = 0

        # * PUT Text
        cvzone.putTextRect(frame, f"Blink Count : {blinkCount}", (100, 100))
        #Todo : resize = Custom size window.
        frame = cv2.resize(frame, (540, 700))
        # * Draw a rectangle on the full frame to show the detection area
        cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 255, 0), 2)
        cv2.imshow('Eye Detewction', frame)


        # Todo : Plot Left
        imgPlot = plotY.update(ratioAvg)
        cv2.imshow("Plot Rigth eye",imgPlot) 
    else:
        face_detected = False # set face detected to false
        cvzone.putTextRect(frame, f"No Face Detected", (100, 100))
        frame = cv2.resize(frame, (700, 540))
        cv2.imshow('Eye Detewction', frame)
        
        if not face_detected:
            listRatioRigth = []
            listRatioRigthRight = []
            counter = 0
            ratioAvg = 0
            ratioAvgRight = 0

    # ออกจากลูปเมื่อกด 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
