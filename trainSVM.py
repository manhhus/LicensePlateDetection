import os
import pickle
import cv2
import numpy as np
import glob
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# kích thước ảnh kỹ tự
digit_w = 30
digit_h = 60

def get_digit_data(path):

    digit_list = []
    label_list = []

    for number in range(10):
        print(path + str(number))
        i=0
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            # đọc ảnh
            img = cv2.imread(img_org_path, 0)
            #chuyển ảnh thành mảng
            img = np.array(img)
            # chỉnh lại kích thước cho đúng tiêu chuản
            img = img.reshape(digit_w,digit_h)
            # thêm ảnh vào trong digit_list
            digit_list.append(img)
            # thêm tên của ký tự vào label_list
            label_list.append([int(number)])

    for number in range(65, 91):
        # chuyển số thành ký tự
        number = chr(number)
        print(path + str(number))
        i=0
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            # đọc ảnh
            img = cv2.imread(img_org_path, 0)
            # chuyển ảnh thành mảng
            img = np.array(img)
            # chỉnh lại kích thước mảng
            img = img.reshape(digit_w,digit_h)
            # thêm ảnh vào trong digit_list
            digit_list.append(img)
            # thêm tên của ký tự vào label_list
            label_list.append([ord(number)])

    return  digit_list, label_list

#lấy dữ liệu
path_train = "datatrain/"
path_test = "datatest/"

# lấy list các ký tự và tên của chúng
digit_train, label_train = get_digit_data(path_train)
digit_test, label_test = get_digit_data(path_test)

# chỉnh sửa lại list
digit_train = np.array(digit_train, dtype=np.float32)
digit_test = np.array(digit_test, dtype=np.float32)

# chỉnh lại kích thước
digit_train = digit_train.reshape(-1, digit_h * digit_w)
digit_test = digit_test.reshape(-1, digit_h * digit_w)

label_train = np.array(label_train)
label_test = np.array(label_test)

# chỉnh lại kích thước
label_train = label_train.reshape(-1, 1)
label_test = label_test.reshape(-1, 1)
# tạo mô hình
model = LinearSVC(C=10)
# fit mô hình
model.fit(digit_train,label_train.ravel())

y_pre = model.predict(digit_test)
print("Score ", accuracy_score(label_test,y_pre))

# xóa file svm.xml trước
#xml = "biensoxe/svm.xml"
#if os.path.exists(xml):
#    os.remove(xml)

# lưu các giá trị đã tính toán vào trong file svm.xml
#pickle.dump(model, open(xml, 'wb'))

