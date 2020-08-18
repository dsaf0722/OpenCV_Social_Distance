# USAGE
# python yolo_real_time.py --output output/airport_output.avi for saving the live video to a file
# python yolo_real_time.py for just live stream

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-o", "--output", default=False,
    help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes) and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# check if the video writer is enabled
if args["output"] is not False:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args["output"], fourcc, 2,(640, 360), True)

(W, H) = (None, None)
print("[INFO] starting video capture...")
cap = VideoStream(src=0).start()
#cap = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

#def known_width
KNOWN_WIDTH_ALL = [
    ["person",19.69],
    ["bottle",3],
    ["tvmonitor",24],
    ["car",78.74],
    ["unknown",10]
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
    ["person",19.69],
    ["bottle",3],
    ["tvmonitor",24],
    ["car",78.74],
    ["unknown",10]
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
IMAGE_PATHS = ["images/Picture5.jpg"]

KNOWN_DISTANCE = 24.0
KNOWN_WIDTH = 11.69

image = cv2.imread(IMAGE_PATHS[0]) 
marker = find_marker(image) 
#     marker = find_marker(frame)           
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
print('focalLength:',focalLength)

# loop over the frames from the video stream


key = cv2.waitKey(1) & 0xFF 

object_list = [
    'person',
    'bottle',
    'tvmonitor',
    'car',
    'unknown'
]

object_count = {obj:0 for obj in object_list}

while True:
    frame = cap.read()
    frame = cv2.resize(frame, (640, 360))

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward pass of the YOLO object detector,
    # giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:     # loop over each of the layer outputs
        for detection in output:        # loop over each of the detections
            # extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if args["output"] is not False:
        writer.write(frame)

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] cleanup up...")
if args["output"] is not False:
    writer.release()
cap.stop()
cv2.destroyAllWindows()
