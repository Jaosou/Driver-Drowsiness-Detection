import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def data_print(x_test,y_test, y_pred):
    if y_test != y_pred:
        print(f"X Test : {x_test}\nY Test : {y_test}\nY Pred : {y_pred}\n")

# โหลดข้อมูลจาก CSV
data = pd.read_csv('C:/Project/End/Code/Data/train/round/all_data_round.csv')
data_test = pd.read_csv('C:/Project/End/Code/Data/train/round/ear_data_for_train_round3.csv')

# แปลงค่า 'mode_eyes' เป็น 0 (Open) และ 1 (Closed)
label_encoder = LabelEncoder()
data['mode_eyes'] = label_encoder.fit_transform(data['mode_eyes'].replace(
    {
        'Open': 0,
        'Close': 1
    }
))

# ใช้ 'ear_value_left' และ 'ear_value_right' เป็น Features
X = data[['ear_value_left', 'ear_value_right']].values
# ใช้ 'mode_eyes' เป็น Label (0 = Open, 1 = Closed)
Y = data['mode_eyes'].values


data_test['mode_eyes'] = label_encoder.fit_transform(data_test['mode_eyes'].replace(
    {'Open': 0, 'Close': 1}
))
x_test_data_input = data_test[['ear_value_left', 'ear_value_right']].values
y_test_data = data_test['mode_eyes'].values

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# สร้างโมเดล SVM
model = SVC(kernel='linear')

# ฝึกโมเดล
model.fit(X_train, y_train)

# ทำนายผลลัพธ์
y_pred = model.predict(x_test_data_input)

list(map(data_print, x_test_data_input,y_test_data, y_pred))

# ประเมินผล
accuracy = accuracy_score(y_test_data, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# หากต้องการประเมินผลที่ละเอียดยิ่งขึ้น
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion Matrix:")
print(confusion_matrix(y_test_data, y_pred))
print("Classification Report:")
print(classification_report(y_test_data, y_pred))

# บันทึกโมเดล
import joblib
joblib.dump(model, 'svm_model.joblib')  
