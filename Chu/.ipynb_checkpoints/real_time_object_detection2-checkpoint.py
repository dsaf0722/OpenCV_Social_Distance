# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import math
from simplewebcam import head_detection

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False, default='category/MobileNetSSD_deploy.prototxt.txt',
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False, default='category/MobileNetSSD_deploy.caffemodel',
    help="path to Caffe pre-trained model")
ap.add_argument("-o", "--txt", required=False, default='head/o.prototxt',
    help="path to head detection prototxt file")
ap.add_argument("-n", "--head", required=False, default='head/onet_iter_90000.caffemodel',
    help="path to head detection model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probabilit to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
 
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# use head_detection model
# net2 = cv2.dnn.readNetFromCaffe(args["txt"], args["head"])

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# time.sleep(2.0)
fps = FPS().start()

#def known_width
KNOWN_WIDTH_ALL = [
    ["person",16],
    ["bottle",2.36],
    ["tvmonitor",21],
    ["car",78.74],
    ["all",10]
]
# 找到目標函式
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

def find_classes(x):
    for i in KNOWN_WIDTH_ALL:
        if x == i[0]:
            return i[1]
    else:
        return KNOWN_WIDTH_ALL[-1][1]
    
def change_object(key):
    KNOWN_WIDTH_ALL = [
    ["person",16],
    ["bottle",2.36],
    ["tvmonitor",21],
    ["car",78.74],
    ["all",10]
]
    if key == ord("p"):
        lst = [KNOWN_WIDTH_ALL[0]]
        return lst
    elif key == ord("b"):
        lst = [KNOWN_WIDTH_ALL[1]]
        return lst
    elif key == ord("t"):
        lst = [KNOWN_WIDTH_ALL[2]]
        return lst
    elif key == ord("c"):
        lst = [KNOWN_WIDTH_ALL[3]]
        return lst
    elif key == ord("a"):
        lst = KNOWN_WIDTH_ALL
        return lst
    else:
        return 0

# use image to Calculation focalLength
IMAGE_PATHS = ["images/U11_1.jpg"]# U11 CAM
# IMAGE_PATHS = ["images/webcam3.jpg","images/webcam4.jpg"]# WEBCAM :focalLength = 1023(26mm)
KNOWN_DISTANCE = 24.0
KNOWN_WIDTH = 11.69

image = cv2.imread(IMAGE_PATHS[0]) 
marker = find_marker(image) 
#     marker = find_marker(frame)           
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH #11XX
focalLength = 1023
print('focalLength:',focalLength)
# 585.7313692926647

# loop over the frames from the video stream

key = cv2.waitKey(1) & 0xFF 

object_list = ['person','bottle','tvmonitor','car','all']
object_count = {obj:0 for obj in object_list}

while True:
# grab the frame from the threaded video stream and resize it
# to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    
#     cv2.imshow("Frame", frame)   
#     key = cv2.waitKey(1) & 0xFF  
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
#     print(blob[0])
#     print(blob.shape)
   
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward() 
#     print(detections)
#     print(detections.shape)
    # loop over the detections
    object_count = {obj:0 for obj in object_list}
    # To count box 
    boxes = []

    for i in np.arange(0, detections.shape[2]):
        # draw a bounding box around the image and display it
        #box = np.int0(cv2.cv.BoxPoints(marker))
#         box = cv2.boxPoints(marker)
#         box = np.int0(box)
#         cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            KNOWN_WIDTH_ALL = KNOWN_WIDTH_ALL if change_object(key) == 0 else change_object(key)
            
            cv2.putText(frame,f'detect {KNOWN_WIDTH_ALL[-1][0]}:',(20, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            for lst in KNOWN_WIDTH_ALL:
                if CLASSES[idx] in lst:
                    object_count[CLASSES[idx]] += 1
    #                 print(count_object(CLASSES[idx]))
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
    #             label = "{}: {:.2f}%".format(CLASSES[idx],
    #                 confidence * 100)

                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
    #             cv2.rectangle(frame, (startX, startY), (endX, endY),
    #                 COLORS[:, 0], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15

                # show box_class
                    cv2.putText(frame,CLASSES[idx],(startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    x = CLASSES[idx] 
                    KNOWN_WIDTH = find_classes(x)
    
    #             worse method to find pix_width 
    #                 marker = find_marker(frame)
    #                 print('marker',marker[1][0])
    #                 if marker == 0:
    #                     print(marker)
    #                     continue
                    box_width_before_standardize = np.abs(startX-endX)
                    boxcenter_to_framecenter = np.abs(((startX+endX)/2)-400)
                    # if box_center_x = 400, box_width_before_standardize = 110 and 
                    # if box_end_x = 800, shrink_box_width = 150
                    # so find the box_width after box_center_x change and reply it
                    # >>>  shrink_box_width = 4/11*box_width_before_standardize*boxcenter_to_framecenter/400+box_width
                    # >>>  box_width = shrink_box_width - 4/11*box_width_before_standardize*boxcenter_to_framecenter/400
                    biggest_shrink = 4/11 * box_width_before_standardize
                    box_width = box_width_before_standardize - biggest_shrink * boxcenter_to_framecenter/400 
#                     print(box_width_before_standardize,box_width)
                    box_height = np.abs(startY-endY)
#                     print(box_width)
                    if box_width == 0:   
                        continue
                    '''distance_to_camera(knownWidth * focalLength) / perWidth''' 
                    inches = distance_to_camera(KNOWN_WIDTH, focalLength,box_width)# marker[1][0])
#                     print(inches,box_width,startX,endX)
                    # find center of box and detect real distance of two object or person
                    # shrink: box_width(inches/100) / obj_width
                    shrink = KNOWN_WIDTH_ALL[0][1]/box_width  
                    # put features in boxes
                    boxes.append([CLASSES[idx],startX,startY,box_width,box_height,inches,endX,endY,shrink])
                    
                    # find distance between two objects
                    for count in boxes:
                        horizontal_distance = 0
                        real_horizontal_distance = 0
                        # function change_object return lst ==> ex:[['person',16]]
                        if CLASSES[idx]== KNOWN_WIDTH_ALL[-1][0] and len(boxes) >= 2:
#                             print(boxes[0][1],boxes[1][1])
                            shrink_ = boxes[0][-1] if boxes[0][-1] < boxes[1][-1] else boxes[1][-1]                        
                            # define center(x,y) of two objects 
                            center1_x = (boxes[0][1]+boxes[0][-3])/2 
                            center2_x = (boxes[1][1]+boxes[1][-3])/2
                            center1_y = (boxes[0][2]+boxes[0][-2])/2
                            center2_y = (boxes[1][2]+boxes[1][-2])/2

                            center_horizontal_distance = np.abs(center1_x - center2_x)
                            # convert to cm
#                             print(f'{center_horizontal_distance * shrink_ * 30.48 / 12} cm')
                            # frame_width = 800 , x_center = 400
                            x1_center = np.abs(400 - center1_x) 
                            x2_center = np.abs(400 - center2_x)
                            # x1_center+center_x2 = center_horizontal_distance
                            # print(x1_center+center_x2,center_horizontal_distance)
                            # convert x1_center and x2_center to real_distance
                            x1_center = x1_center * shrink_
                            x2_center = x2_center * shrink_
#                             print(x1_center,x2_center)

                            # find sin(y1+y2) # sin(y1) = x1_center/d1*sin(90')
                            if x1_center == 0 and x2_center != 0:
                                A1 = 90
                                A2 = 90 - np.abs(math.acos(x2_center/boxes[1][-4])*180/math.pi)
                            elif x2_center == 0 and x1_center != 0:
                                A1 = 90 - np.abs(math.acos(x1_center/boxes[0][-4])*180/math.pi) 
                                A2 = 90
                            elif x1_center != 0 and x2_center != 0:
                                A1 = 90 - np.abs(math.acos(x1_center/boxes[0][-4])*180/math.pi) 
                                A2 = 90 - np.abs(math.acos(x2_center/boxes[1][-4])*180/math.pi)
#                             print(A1,A2)
                            if center1_x < 400 and center2_x < 400:
                                A = np.abs(A1-A2)
                            elif center1_x > 400 and center2_x > 400:
                                A = np.abs(A1-A2)
                            else:
                                A = A1 + A2
                            A_ = A * math.pi / 180
#                             print(A)
                            # cos(A)--error, cos(A_)--correct
#                             real_distance_of_objects = np.abs((boxes[0][-4])**2 + (boxes[1][-4])**2 - (2*(boxes[0][-4])*(boxes[1][-4]) * math.cos(A)))**0.5*30.48/12
                            real_distance_of_objects2 = np.abs(((boxes[0][-4])**2 + (boxes[1][-4])**2 - (2*(boxes[0][-4])*(boxes[1][-4]) * math.cos(A_)))**0.5*30.48/12)*0.01
        
#                             print((boxes[0][-4]),(boxes[1][-4]),real_distance_of_objects,real_distance_of_objects2)
                            line_start_x = center1_x if center1_x < center2_x else center2_x
                            line_start_x = int(line_start_x)
                            line_end_x = center2_x if center2_x > center1_x else center1_x
                            line_end_x = int(line_end_x)
                            line_start_y = center1_y if center1_x < center2_x else center2_y
                            line_start_y = int(line_start_y) 
                            line_end_y = center2_y if center2_x > center1_x else center1_y
                            line_end_y = int(line_end_y)

                            cv2.line(frame,(line_start_x,line_start_y),(line_end_x,line_end_y),(0,0,200),2)
                            cv2.putText(frame,'%.2f m' % real_distance_of_objects2,
                                    (line_start_x+100,line_start_y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                                 1.0,(0,0,200),2)
#                             cv2.putText(frame,'%.2f cm' % real_distance_of_objects2,
#                                     (line_start_x+100,line_start_y-10), cv2.FONT_HERSHEY_SIMPLEX,
#                                                  1.0,(0,0,200),2)

                    cv2.putText(frame, "%.2fcm" % (inches*30.48/12),(startX+100, y), 
                cv2.FONT_HERSHEY_SIMPLEX,1.0,COLORS[idx], 2)
            cv2.putText(frame, f'{object_count}',(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)   
    key = cv2.waitKey(50) & 0xFF 

    # if the ` ` key was pressed, stop the frame
    if key == ord(" "):
        cv2.waitKey(0)
    # use esc to close window 
    head_width = []
    if key == ord("z"):
#         cv2.destroyAllWindows()
        switch = 1
        head_width.append(head_detection(switch))
        print(head_width)
        if key == 27:   
            cv2.imshow("Frame", frame)   

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
     # update the FPS counter
    fps.update()       

 # stop the timer and display FPS information
# fps.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
