from tensorflow.keras import models, layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# โหลดข้อมูลจาก CSV
data = pd.read_csv('C:/Project/End/Code/Data/train/round/all_data_round.csv')
data_test = pd.read_csv('C:/Project/End/Code/Data/train/round/ear_data_for_train_round2.csv')

# เตรียมข้อมูล
X = data[['ear_value_left', 'ear_value_right']].values
X = (X - X.min()) / (X.max() - X.min())  # Normalize ข้อมูล

y = data['mode_eyes'].map({'Open': 0, 'Closed': 1}).values  # แปลง 'Open' -> 0, 'Closed' -> 1

X_test_data = data_test[['ear_value_left', 'ear_value_right']].values
y_test_data = data_test['mode_eyes'].map({'Open': 0, 'Closed': 1}).values

# แบ่งข้อมูลเป็นชุดเทรนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล MLP
model = models.Sequential()
model.add(layers.Dense(128, input_dim=2, activation='relu'))  # Dense layer แรก

model.add(layers.Dense(64, activation='relu'))  # เลเยอร์ที่ 2
model.add(layers.Dense(32, activation='relu'))   # เลเยอร์ที่ 3# Dense layer ที่สอง

model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))  # ใช้ sigmoid สำหรับการทำนาย 2 คลาส (เปิดตา/หลับตา)

# คอมไพล์โมเดล
model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0,learning_rate=0.00001),
              loss='binary_crossentropy', metrics=['accuracy'])



# ฝึกโมเดล
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test_data, y_test_data))

model.summary()

# ประเมินโมเดล
test_loss, test_acc = model.evaluate(X_test_data, y_test_data)
print(f"Test accuracy: {test_acc}")
