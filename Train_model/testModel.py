import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# โหลดข้อมูลจาก CSV
data_almond = pd.read_csv('C:/Project/End/Code/Data/train/almond/all_data_almond.csv')
data_round = pd.read_csv('C:/Project/End/Code/Data/train/round/all_data_round.csv')

data = data_almond

# การเตรียมข้อมูล
# เราจะใช้ 'ear_value_left' และ 'ear_value_right' เป็นฟีเจอร์ และ 'mode_eyes' เป็น label
X = data[['ear_value_left', 'ear_value_right']].values
y = data['mode_eyes'].map({'Open': 0, 'Closed': 1}).values  # เปลี่ยน 'Open' -> 0, 'Closed' -> 1

# การปรับขนาดข้อมูล
X = X.astype('float32') / 1.0  # Normalize ข้อมูล

# แบ่งข้อมูลเป็นชุดเทรนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape ข้อมูลให้เหมาะสมกับ CNN (เปลี่ยนเป็น 2D image)
X_train = X_train.reshape(-1, 1, 2, 1)  # (จำนวนตัวอย่าง, สูง, กว้าง, ช่องสี)
X_test = X_test.reshape(-1, 1, 2, 1)

# สร้างโมเดล CNN
model = models.Sequential()

# เพิ่มเลเยอร์ Conv2D และ MaxPooling
model.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=(1, 2, 1)))  # ใช้ kernel ขนาด (1, 1)
model.add(layers.MaxPooling2D((1, 1)))  # MaxPooling

model.add(layers.Conv2D(64, (1, 1), activation='relu'))  # ใช้ kernel ขนาด (1, 1) อีกครั้ง
model.add(layers.MaxPooling2D((1, 1)))  # MaxPooling

# เพิ่มเลเยอร์ GlobalAveragePooling2D เพื่อให้ผลลัพธ์มีขนาดที่เหมาะสม
model.add(layers.GlobalAveragePooling2D())

# เพิ่ม Dense layer เพื่อการทำนาย
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid สำหรับการทำนาย 2 คลาส

# สรุปโมเดล
model.summary()

# คอมไพล์โมเดล
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# เทรนโมเดล
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# ประเมินโมเดล
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# บันทึกโมเดล
model.save('almond_model.h5')
