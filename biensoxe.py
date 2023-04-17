import os
import imutils
import cv2
import numpy as np
import glob

def prepocessing(img):
    # chuyển hình ảnh từ ảnh RGB sang ảnh xám
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Giảm nhiễu bằng phương pháp lọc song phương
    noise_removal = cv2.bilateralFilter(gray_img,12,30,30)
    #cân bằng histogram
    equal_histogram = cv2.equalizeHist(noise_removal)
    # tạo nhân 3*3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # dùng thuật toán mở loại bỏ nhiễu với ảnh đầu vào là ảnh cân bằng, và nhân kích cở 3 * 3
    morphology_img = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel)
    # tìm biên ảnh bằng phương pháp Canny với đầu vào là ảnh đã qua xóa nhiễu
    edged_img = cv2.Canny(morphology_img ,30,200)
    return edged_img

def findContours(edged_img):
    contours, hierarchy = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sắp xếp các contour có kích thước lớn lên đầu và chỉ lấy 10 phần tử
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]
    # đưa vào list những contour có hình dạng tứ giác
    listContours = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        # nếu số đỉnh bằng 4 thì ta thêm vào list
        if len(approx) == 4:
            listContours.append(approx)
    return listContours

def characterSegment(image,lenCon):
    # chuyển ảnh đầu vào thành ảnh xám
    roi_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # làm mờ hình ảnh bằng Gausianblur với đầu vào là ảnh xám
    roi_blur = cv2.GaussianBlur(roi_gray, (9,9),1)
    # nhịn phân hóa ảnh
    ret, thre = cv2.threshold(roi_blur, 100,255,cv2.THRESH_BINARY_INV)
    # sắp xếp contour theo thứ tự từ trái sang phải
    kernel = np.ones((3,3), np.uint8)
    # nở ảnh để bù nét liền bị 
    thre = cv2.dilate(thre,kernel,iterations=1)
    # tìm countours chứa chữ trên biển
    cont, hier = cv2.findContours(thre,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sắp xếp các contour có kích thước lớn lên đầu và chỉ lấy 10 phần tử
    cont = sorted(cont,key=cv2.contourArea,reverse=True)[:lenCon]
    # trả về contours và ảnh nhị phân
    return cont,thre

def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts   

def detectCharacter(cropped,model_svm,lenCon):
    # ta lấy ra list contour và ảnh nhị phân của ảnh biển số xe thông qua phương thức characterSegment 
    listContour,thre = characterSegment(cropped,lenCon)
    plate_info = ""
    # vẽ contours của các ký tự trong ảnh biển số xe
    for cnt in sort_contours(listContour):
        x,y,w,h = cv2.boundingRect(cnt)
        # lấy tỉ lệ của chiều dài và chiều rộng
        ratio = h/w
        # kiểm tra điều kiện nếu tỉ lệ mà nhỏ hơn 1.2 (quá vuông) hoặc lớn hơn 4 (quá cao) thì không lấy
        if 1.2<=ratio<=4:
            cv2.rectangle(cropped,(x,y),(x+w,y+h),(0,255,0),2)
            # cắt ra các ký tự con của biển số xe
            child_char = thre[y:y+h,x:x+w]
            # chỉnh lại kích thước ảnh ký tự con với hình dạng theo digit_w, và digit_h
            child_char = cv2.resize(child_char, dsize=(30, 60))
            # chuyển ảnh ký tự thành ảnh nhị phân 
            #_, child_char = cv2.threshold(child_char, 127, 255, cv2.THRESH_BINARY)
            # chuyển ảnh ký tự thành mảng
            child_char = np.array(child_char,dtype=np.float32)
            # chỉnh chửa lại ảnh ký tự theo hình dạng dưới
            child_char = child_char.reshape(-1, 30 * 60)
            # đưa ảnh ký tự vào svm model và tính toán, giá trị trả về là 1 con số
            result = model_svm.predict(child_char)[1]
            result = int(result[0, 0])
            if result <= 9: # Nếu là số thì hiện số
                result = str(result)
            elif result >= 65 and result < 91: # Nếu là chữ thì chuyển qua bảng ASCII
                result = chr(result)
            # thêm ký tự vào kết quả sẽ hiện ra
            plate_info +=result
    return plate_info

def main():
    listImage = []

    # findPaths = glob.glob("anh/*.jpg")
    # for findPath in findPaths:
    #     img = cv2.imread(findPath)
    #     listImage.append(img)
    
    img = cv2.imread("anh/2.jpg")
    listImage.append(img)
    
    for img in listImage:
        # ta chỉnh kích thước của ảnh với chiều rộng là 800 và chiều dài là auto
        img = imutils.resize(img, width=800 )
        imgPre = prepocessing(img)
        # ta thực hiện lấy list contour thông qua phương thức processing
        list = findContours(imgPre)

        # ta lấy giá vị x,y trí bắt đầu của contour lớn nhất, và giá trị chiều rộng chiều dài của nó
        x, y ,w,h = cv2.boundingRect(list[0])
        # lấy ảnh biển số xe thông qua ảnh ban đầu
        cropped = img[y+4:y+h-2,x+3:x+w+2]

        # ta chỉnh kích thước của ảnh biển số xe với chiều rộng là 500 và chiều dài là auto
        cropped = imutils.resize(cropped, width=500 )

        cv2.drawContours(img,list,0,(0,255,0),4)

        # hiển thị ảnh biển số xe
        cv2.imshow("anh bien", cropped)
        cv2.waitKey(0)

        # tải mô hình svm đã train với dữ liệu đã gán nhãn ở trên
        model_svm = cv2.ml.SVM_load('svm.xml')
        
        plate_info = ""
        ratio = w/h
        # nếu tỉ lệ biển sấp xỉ hình vuông thì ta chia biển thành 2 nửa trên dưới để nhận diện
        if 0.8<=ratio<=2.8:
            h,w,_ = cropped.shape
            half = h // 2
            top = cropped[:half, :]
            bot = cropped[half: , :]
            plate_info += detectCharacter(top,model_svm,4)
            plate_info += detectCharacter(bot,model_svm,6)
        else:
            plate_info += detectCharacter(cropped,model_svm,10)

        # hiển thị ảnh biển số xe sau khi đã vẽ contour
        cv2.imshow("anh bien ve contour",cropped)
        cv2.waitKey(0)
        
        # hiện ký tự ra terminal
        print("Bien so=", plate_info)
main()
