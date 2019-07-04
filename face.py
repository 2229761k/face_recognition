import numpy as np
import pandas as pd
import cv2
# from libsvm import *
from PIL import Image
import glob
from matplotlib import pyplot as plt
# dataset
# train sets
train_set_ID_1 = []
train_set_ID_2 = []
train_set_ID_3 = []
train_set_ID_4 = []
train_set_ID_5 = []
train_set_ID_6 = []
train_set_ID_7 = []
path = ['ID_1', 'ID_2', 'ID_3', 'ID_4','ID_5','ID_6','ID_7']

for i in range(7):
    for img in glob.glob('/home/doh/Desktop/workspace/assignment1/Train/' + path[i] +'/*.jpg'):
        if path[i] == 'ID_1':
            train_set_ID_1.append(img)
        if path[i] == 'ID_2':
            train_set_ID_2.append(img)
        if path[i] == 'ID_3':
            train_set_ID_3.append(img)
        if path[i] == 'ID_4':
            train_set_ID_4.append(img)
        if path[i] == 'ID_5':
            train_set_ID_5.append(img)
        if path[i] == 'ID_6':
            train_set_ID_6.append(img)
        if path[i] == 'ID_7':
            train_set_ID_7.append(img)

# test sets
test_set_ID_1 = []
test_set_ID_2 = []
test_set_ID_3 = []
test_set_ID_4 = []
test_set_ID_5 = []
test_set_ID_6 = []
test_set_ID_7 = []
path = ['ID_1', 'ID_2', 'ID_3', 'ID_4','ID_5','ID_6','ID_7']

for i in range(7):
    for img in glob.glob('/home/doh/Desktop/workspace/assignment1/Test/' + path[i] +'/*.jpg'):
        if path[i] == 'ID_1':
            test_set_ID_1.append(img)
        if path[i] == 'ID_2':
            test_set_ID_2.append(img)
        if path[i] == 'ID_3':
            test_set_ID_3.append(img)
        if path[i] == 'ID_4':
            test_set_ID_4.append(img)
        if path[i] == 'ID_5':
            test_set_ID_5.append(img)
        if path[i] == 'ID_6':
            test_set_ID_6.append(img)
        if path[i] == 'ID_7':
            test_set_ID_7.append(img)
# print('----------------------')
# print(test_set_ID_1[0])
# print('----------------------')

# using openCV 
face_cascade = cv2.CascadeClassifier('/home/doh/Desktop/workspace/assignment1/xmls/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/doh/Desktop/workspace/assignment1/xmls/haarcascade_eye.xml')

img = cv2.imread(test_set_ID_1[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


# second step -> nomarlize

normalizedImg = cv2.resize(img, (500, 500))
normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)

# print(img)
# print(img[0])

# ------------------LBP--------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# comparing with center value
def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    
    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top

    # don't understand this part
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color = "black")
            current_plot.set_xlim([0,260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)            
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)

    plt.show()

height, width, channel = img.shape

img_lbp = np.zeros((height, width,3), np.uint8)
for i in range(0, height):
    for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(gray, i, j)
hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
output_list = []
output_list.append({
    "img": gray,
    "xlabel": "",
    "ylabel": "",
    "xtick": [],
    "ytick": [],
    "title": "Gray Image",
    "type": "gray"        
})
output_list.append({
    "img": img_lbp,
    "xlabel": "",
    "ylabel": "",
    "xtick": [],
    "ytick": [],
    "title": "LBP Image",
    "type": "gray"
})    
output_list.append({
    "img": hist_lbp,
    "xlabel": "Bins",
    "ylabel": "Number of pixels",
    "xtick": None,
    "ytick": None,
    "title": "Histogram(LBP)",
    "type": "histogram"
})

show_output(output_list)



cv2.imshow('img', normalizedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# -----------------------------------------------Nearest Neighbor---------------------------------------------
class NearestNeighbor:
    def __init__(self):
        pass

    def nn_train(self, train_img, test_img):
        distance = np.sqrt(np.sum(np.square(train_img - test_img)))
        return distance

    def nn_predict(self, train_img):


    




# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

