import cv2

# โหลด Haar Cascade สำหรับการตรวจจับหน้าและดวงตา
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# เริ่มต้นกล้อง
cap = cv2.VideoCapture(0)

while True:
    # อ่านกรอบภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # แปลงภาพเป็นขาวดำ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # ตรวจจับดวงตาในแต่ละใบหน้า
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # วาดกรอบรอบใบหน้า
        roi_gray = gray[y:y + h, x:x + w]  # พื้นที่ที่ตรวจจับใบหน้า
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # วาดกรอบรอบดวงตา

    # แสดงผลภาพ
    cv2.imshow('Eye Detewction', frame)

    # ออกจากลูปเมื่อกด 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
