# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


cap = cv2.VideoCapture(0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
#　使用opencv預設的SVM分類器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def find_marker(frame):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    gray = cv2.GaussianBlur(gray, (5, 5), 0)        
    edged = cv2.Canny(gray, 35, 125)                
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
    #(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 求最大面積 
    c = max(cnts, key = cv2.contourArea)
 
    # compute the bounding box of the of the paper region and return it
    # cv2.minAreaRect() c代表點集，返回rect[0]是最小外接矩形中心點座標，
    # rect[1][0]是width，rect[1][1]是height，rect[2]是角度
    return cv2.minAreaRect(c)

# 距離計算函式 
def distance_to_camera(knownWidth, focalLength, perWidth):  
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth 

while(1):
    # get a frame
    ret, frame = cap.read()
    
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))


    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
         padding=(8, 8), scale=1.01)
    
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    #　非極大抑制 消除多餘的框 找到最佳人體
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    # 畫出邊框
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera = cv2.VideoCapture(0)

while camera.isOpened():
    # get a frame
    grabbed, frame = camera.read()
    marker = find_marker(frame)
    if marker == 0:
        print(marker)
        continue
#     inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    
    for (xA, yA, xB, yB) in marker:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    xa_max = xA
    xb_max = xB

    pix_width = xb_max - xa_max
    if pix_width == 0:
    #pix_height = 1
        continue
    print (pix_height)
    #print (pix_height)
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, pix_width)
    print("%.2fcm" % (inches *30.48/ 12))

    # draw a bounding box around the image and display it
    #box = np.int0(cv2.cv.BoxPoints(marker))
    box = cv2.boxPoints(marker)
    box = np.int0(box)
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

    # inches 轉換為 cm
    cv2.putText(frame, "%.2fcm" % (inches *30.48/ 12),
             (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
         2.0, (0, 255, 0), 3)

    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
