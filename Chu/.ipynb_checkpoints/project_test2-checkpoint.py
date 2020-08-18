#import the necessary packages
import numpy as np
import cv2

# 找到目標函式
def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
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
 
# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0

# initialize the known object width, which in this case, the piece of
# paper is 11 inches wide
# A4紙的長和寬(單位:inches)
KNOWN_WIDTH = 11.69
# KNOWN_HEIGHT = 8.27

# initialize the list of images that we'll be using
IMAGE_PATHS = ["images/0710.jpg"]
 
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
#讀入第一張圖，通過已知距離計算相機焦距QQ
image = cv2.imread(IMAGE_PATHS[0]) 
marker = find_marker(image)           
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH  

#通過攝像頭標定獲取的畫素焦距
#focalLength = 811.82
print('focalLength = ',focalLength)

#開啟攝像頭
camera = cv2.VideoCapture(0)

while camera.isOpened():
    # get a frame
    grabbed, frame = camera.read()
    marker = find_marker(frame)
    if marker == 0:
        print(marker)
        continue
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    
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
camera.release()
cv2.destroyAllWindows() 