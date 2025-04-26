import numpy as np
import pandas as pd
import csv
import os

# อ่านข้อมูลจากไฟล์ CSV
file_path = 'C:/Project/End/Code/ear_data.csv'
data = pd.read_csv(file_path)

range_close_eyes = 0.45


#PATH File CSV
csv_file_name = "ear_data_for_train.csv"
# Check if the CSV file exists
file_exists = os.path.exists(csv_file_name)

with open(csv_file_name, "a", newline="") as csvfile:
    fieldnames = ['person_id', 'eye_type', 'ear_value_left','ear_value_right','mode_eyes']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
    if not file_exists:
        writer.writeheader()


#Todo : Get ID
def get_all_id():
    unique_person_id = pd.to_numeric(data['person_id'].unique(), errors='coerce')
    return unique_person_id


# Find Last ID
def find_last_person_id():
    file_path = 'ear_data.csv'
    data = pd.read_csv(file_path)

    # ดึง person_id จากแถวสุดท้าย
    last_person_id = data['person_id'].iloc[-1] 

    print(f"Last person_id: {last_person_id}")
    return last_person_id

    #Write the data into a CSV file [id,type_of_eyes,left,right,result]
def write_EAR_to_file(data):
    with open(csv_file_name, "a", newline="") as csvfile:
        fieldnames = ['person_id', 'eye_type', 'ear_value_left','ear_value_right','mode_eyes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        print(file_exists)
        
            
        writer.writerow({
            'person_id': data[0],
            'eye_type': data[1],
            'ear_value_left': f"{data[2]:.3f}",
            'ear_value_right': f"{data[3]:.3f}",
            'mode_eyes' : f"{data[4]}"
        })
        print(data)
    print()


#Check Close And Open Eyes
def check_close_eyes(left,right,median_left,median_right,delta,range_close_eyes):
    if left < median_left - delta*range_close_eyes and right < median_right - delta*range_close_eyes :
            print(f"{left} : {right} : Close Eyes\n")
            return "Close"
    else : 
            print(f"{left} : {right} : Open Eyes\n")
            return "Open"


#Check Type of eyes
def check_eyes(max_left,max_right,delta,range_close_eyes,median_left,median_right,data_left,data_right):
    id = find_last_person_id()
    type_of_eyes = "Almond" if max_left < 0.25 and max_right < 0.25 else "Round"
    print(f"{median_left:.3f}")
    print(f"{median_right:.3f}")
    print(f"{median_left-delta*range_close_eyes:.3f}")
    print(f"{median_right-delta*range_close_eyes:.3f}\n")
    
    count_close = 0
    count_open = 0
    
    for left,right in zip(data_left,data_right) :
        result = check_close_eyes(left,right,median_left,median_right,delta,range_close_eyes)
        if result == "Close" : count_close += 1
        else : count_open += 1
        write_EAR_to_file([id,type_of_eyes,left,right,result])
    print(f"Count Close : {count_close}\nCount Open : {count_open}\n")


def calculate_median(max,min):
    return (max + min)/2

# แปลงข้อมูลใน ear_value_left และ ear_value_right เป็นตัวเลข และกรองค่าที่ไม่สามารถแปลงได้
ear_value_left = pd.to_numeric(data['ear_value_left'], errors='coerce')
ear_value_right = pd.to_numeric(data['ear_value_right'], errors='coerce')

# กรอง NaN ออก (ถ้ามีค่า NaN หลังจากการแปลง)
ear_value_left = ear_value_left.dropna()
ear_value_right = ear_value_right.dropna()

# หรือใช้ unique() เพื่อดึงค่าที่ไม่ซ้ำ
unique_left_np = pd.to_numeric(ear_value_left.unique(), errors='coerce')
unique_right_np = pd.to_numeric(ear_value_right.unique(), errors='coerce')

# ใช้ np ในการคำนวณค่าเฉลี่ย, ค่ากลาง, ค่าสูงสุด, ค่าต่ำสุด
mean_left = np.mean(ear_value_left)
mean_right = np.mean(ear_value_right)

# ใช้ pandas เพื่อหาจำนวนค่าที่ซ้ำกันในคอลัมน์ ear_value_left และ ear_value_right
count_left = data['ear_value_left'].value_counts()
count_right = data['ear_value_right'].value_counts()

top_3_left = np.partition(unique_left_np,-3)[-3:]
top_3_right = np.partition(unique_right_np,-3)[-3:]

min_3_left = np.partition(unique_left_np,3)[:3]
min_3_right = np.partition(unique_right_np,3)[:3]

weights_top_left = {value : count_left.get(top_3_left[0],0) for value in top_3_left}
weights_top_right = {value : count_right.get(top_3_right[0],0) for value in top_3_right}

weights_smallest_left = {value : count_left.get(min_3_left[0],0) for value in min_3_left}
weights_smallest_right = {value : count_right.get(min_3_right[0],0) for value in min_3_right}

weights_top_left_list = list(weights_top_left.values())
weights_top_right_list = list(weights_top_right.values())  
weights_smallest_left_list = list(weights_smallest_left.values())
weights_smallest_right_list = list(weights_smallest_right.values())

# คำนวณค่าเฉลี่ยถ่วงน้ำหนัก (Weighted Average)
weighted_avg_top_left = np.average(top_3_left, weights=weights_top_left_list)
weighted_avg_top_right = np.average(top_3_right, weights=weights_top_right_list)

weighted_avg_smallest_left = np.average(min_3_left, weights=weights_smallest_left_list)
weighted_avg_smallest_right = np.average(min_3_right, weights=weights_smallest_right_list)

median_left = calculate_median(weighted_avg_top_left,weighted_avg_smallest_left)
median_right = calculate_median(weighted_avg_top_right,weighted_avg_smallest_right)

delta_ear_left = weighted_avg_top_left - median_left
delta_ear_right = weighted_avg_top_right - median_right

# แสดงผลลัพธ์
print(f"\n==================================\nMax left : \n{top_3_left}\nMax right : \n{top_3_right}\n==================================")

print(f"\n==================================\nMin left : \n{min_3_left}\nMin right : \n{min_3_right}\n==================================")

print(f"\n==================================\nWeight Max left : \n{weights_top_left}\nWeight Max right : \n{weights_top_right}\n==================================")

print(f"\n==================================\nWeight Min left : \n{weights_smallest_left}\nWeight Min right : \n{weights_smallest_right}\n==================================")

print(f"\n==================================\nAvg Max left : \n{weighted_avg_top_left:.3f}\nAvg Max right : \n{weighted_avg_top_right:.3f}\n==================================")

print(f"\n==================================\nAvg Min left : \n{weighted_avg_smallest_left:.3f}\nAvg Min right : \n{weighted_avg_smallest_right:.3f}\n==================================")

print(f"\n==================================\nMedian left : \n{median_left:.3f}\nMedian right : \n{median_right:.3f}\n==================================")

print(f"\n==================================\nDelta left : \n{delta_ear_left:.3f}\nDelta right : \n{delta_ear_right:.3f}\n==================================")

print(f"\n==================================\nDelta left {range_close_eyes} : \n{(delta_ear_left*range_close_eyes):.3f}\nDelta right {range_close_eyes} : \n{(delta_ear_right*range_close_eyes):.3f}\n==================================")

find_last_person_id()

check_eyes(weighted_avg_top_left,weighted_avg_top_right,delta_ear_left,range_close_eyes,median_left,median_right,unique_left_np,unique_right_np)