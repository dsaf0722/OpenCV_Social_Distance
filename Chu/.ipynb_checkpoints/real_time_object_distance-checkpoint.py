# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
# CLASSES = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
 
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()

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

IMAGE_PATHS = ["images/U11_1.jpg"]

# IMAGE_PATHS = ["images/Picture1.jpg"]#, "images/Picture2.jpg", "images/Picture3.jpg"]



# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (48, 48)),
        0.007843, (48, 48), 127.5)
#     print(blob[0])
    print(blob.shape)
#     blob2 = cv2.dnn.blobFromImage(cv2.resize(frame, (100, 100)),
#         0.007843, (300, 300), 127.5)
#     print(blob)
    #image distance for Picture1
    KNOWN_DISTANCE = 24.0
    # A4 width
    KNOWN_WIDTH = 11.69
    
    image = cv2.imread(IMAGE_PATHS[0]) 
    marker = find_marker(image) 
#     marker = find_marker(frame)           
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    print(detections)
#     print(detections.shape)
           
    # loop over the detections
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
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #count distance
#             focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH            
            marker = find_marker(frame)
            if marker == 0:
                print(marker)
                continue
            objecct_pix_width = endX - startX
#             print(objecct_width)
            print()
            print(marker[1][0],objecct_pix_width)
            inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

            # draw the prediction on the frame
#             label = "{}: {:.2f}%".format(CLASSES[idx],
#                 confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
#             cv2.rectangle(frame, (startX, startY), (endX, endY),
#                 COLORS[:, 0], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame,CLASSES[idx],(startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            # inch transform cm
            cv2.putText(frame,"%.2fcm" % (inches * 30.48/ 12),(startX+100, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
#             cv2.putText(frame, "%.2fcm" % (inches *30.48/ 12),(startX, y), cv2.FONT_HERSHEY_SIMPLEX,
#              1.0, (0, 255, 0), 2)
    
    # show the output frame
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
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
