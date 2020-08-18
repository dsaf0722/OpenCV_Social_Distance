# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import math
from flask import Flask, render_template, Response

def Chu():
    cv2.nameWindow = ('Frame',cv2.WINDOW_NORMAL)
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False, default='Chu/category/MobileNetSSD_deploy.prototxt.txt',
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=False, default='Chu/category/MobileNetSSD_deploy.caffemodel',
        help="path to Caffe pre-trained model")
    ap.add_argument("-o", "--txt", required=False, default='Chu/head/o.prototxt',
        help="path to head detection prototxt file")
    ap.add_argument("-n", "--head", required=False, default='Chu/head/onet_iter_90000.caffemodel',
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

    # initialize the video stream, allow the camera sensor to warm up,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    #def known_width
    KNOWN_WIDTH_ALL = [
        ["person",16],
        ["bottle",2.36],
        ["tvmonitor",21],
        ["car",78.74],
        ["all",10]
    ]

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
 # define focalLength
    focalLength = 1023

    # loop over the frames from the video stream
    key = cv2.waitKey(1) & 0xFF 
    object_list = ['person','bottle','tvmonitor','car','all']
    object_count = {obj:0 for obj in object_list}

    while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        frame2 = imutils.resize(frame, width=200,height=200)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)
        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()
        
        # loop over the detections
        object_count = {obj:0 for obj in object_list}
        boxes = []

        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
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
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                    # show box_class
                        cv2.putText(frame,CLASSES[idx],(startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                        x = CLASSES[idx] 
                        KNOWN_WIDTH = find_classes(x)

                        box_width_before_standardize = np.abs(startX-endX)
                        boxcenter_to_framecenter = np.abs(((startX+endX)/2)-400)
                        # if box_center_x = 400, box_width_before_standardize = 110 and 
                        # if box_end_x = 800, shrink_box_width = 150
                        # so find the box_width after box_center_x change and reply it
                        # >>>  shrink_box_width = 4/11*box_width_before_standardize*boxcenter_to_framecenter/400+box_width
                        # >>>  box_width = shrink_box_width - 4/11*box_width_before_standardize*boxcenter_to_framecenter/400
                        biggest_shrink = 4/11 * box_width_before_standardize
                        box_width = box_width_before_standardize - biggest_shrink * boxcenter_to_framecenter/400 
                        box_height = np.abs(startY-endY)
                        
                        if box_width == 0:   
                            continue
                        '''distance_to_camera(knownWidth * focalLength) / perWidth''' 
                        inches = distance_to_camera(KNOWN_WIDTH, focalLength,box_width)# marker[1][0])
                        # find center of box and detect real distance of two object or person
                        # shrink: box_width(inches/100) / obj_width
                        shrink = KNOWN_WIDTH_ALL[0][1]/box_width  
                
                        box_color = COLORS[idx]
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                        box_color, 2)

                        # put features in boxes
                        boxes.append([CLASSES[idx],startX,startY,box_width,box_height,inches,endX,endY,shrink])

                        # find distance between two objects
                        for count in boxes:
                            horizontal_distance = 0
                            real_horizontal_distance = 0
                            # function change_object return lst ==> ex:[['person',16]]
                            if CLASSES[idx]== KNOWN_WIDTH_ALL[-1][0] and len(boxes) >= 2:
                                shrink_ = boxes[0][-1] if boxes[0][-1] < boxes[1][-1] else boxes[1][-1]                        
                                # define center(x,y) of two objects 
                                center1_x = (boxes[0][1]+boxes[0][-3])/2 
                                center2_x = (boxes[1][1]+boxes[1][-3])/2
                                center1_y = (boxes[0][2]+boxes[0][-2])/2
                                center2_y = (boxes[1][2]+boxes[1][-2])/2

                                center_horizontal_distance = np.abs(center1_x - center2_x)
                                # convert to cm
                                # frame_width = 800 , x_center = 400
                                x1_center = np.abs(400 - center1_x) 
                                x2_center = np.abs(400 - center2_x)
                                # x1_center+center_x2 = center_horizontal_distance
                                # convert x1_center and x2_center to real_distance
                                x1_center = x1_center * shrink_
                                x2_center = x2_center * shrink_
                                
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
                                if center1_x < 400 and center2_x < 400:
                                    A = np.abs(A1-A2)
                                elif center1_x > 400 and center2_x > 400:
                                    A = np.abs(A1-A2)
                                else:
                                    A = A1 + A2
                                A_ = A * math.pi / 180
                                
                                # cos(A)--error, cos(A_)--correct
                                real_distance_of_objects2 = np.abs(((boxes[0][-4])**2 + (boxes[1][-4])**2 - (2*(boxes[0][-4])*(boxes[1][-4]) * math.cos(A_)))**0.5*30.48/12)*0.01
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
                                if real_distance_of_objects2 < 1.5:
                                    box_color = (0,0,255)
                                    cv2.rectangle(frame, (boxes[0][1], boxes[0][2]), (boxes[0][-3], boxes[0][-2]),box_color, 2)
                                    cv2.rectangle(frame, (boxes[1][1], boxes[1][2]), (boxes[1][-3], boxes[1][-2]),box_color, 2)
                        cv2.putText(frame, "%.2fm" % (inches*30.48/12*0.01),(startX+100, y), 
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,COLORS[idx], 2)
                cv2.putText(frame, f'{object_count}',(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255), 2)

        # show the output frame
        cv2.imshow("Frame", frame2)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
       
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(1) & 0xFF 

        # if the ` ` key was pressed, stop the frame
        if key == ord(" "):
            cv2.waitKey(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

