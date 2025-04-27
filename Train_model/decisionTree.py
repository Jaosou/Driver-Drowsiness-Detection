from sklearn.calibration import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

def print_data(x_test, y_pred):
    print(f"X Test : {x_test}\nY Pred : {y_pred}\n")

# โหลดข้อมูลจาก CSV
data = pd.read_csv('ear_data_for_train.csv')
data_test = pd.read_csv('ear_data_for_train1.csv')

# แปลงค่า 'mode_eyes' เป็น 0 (Open) และ 1 (Closed)
label_encoder = LabelEncoder()
data['mode_eyes'] = label_encoder.fit_transform(data['mode_eyes'].replace(
    {'Open': 0, 'Close': 1}
))

data_test['mode_eyes'] = label_encoder.fit_transform(data_test['mode_eyes'].replace(
    {'Open': 0, 'Close': 1}
))
x_data_test = data_test[['ear_value_left', 'ear_value_right']].values
y_test_data = data_test['mode_eyes'].values

# ใช้ 'ear_value_left' และ 'ear_value_right' เป็น Features
X = data[['ear_value_left', 'ear_value_right']].values
# ใช้ 'mode_eyes' เป็น Label (0 = Open, 1 = Closed)
y = data['mode_eyes'].values
print(len(x_data_test))

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

print(f"Y : test : {len(y_test)}")

# สร้างโมเดล Decision Tree
model = DecisionTreeClassifier(random_state=42)

# ฝึกโมเดล
model.fit(X_train, y_train)

# ทำนายผลลัพธ์
y_pred = model.predict(x_data_test)

# ใช้ map() เพื่อแทนที่ zip()
list(map(print_data, x_data_test, y_pred))

# ประเมินผล
accuracy = accuracy_score(y_test_data, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# แสดงผล Confusion Matrix และ Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_test_data, y_pred))

print("Classification Report:")
print(classification_report(y_test_data, y_pred))
