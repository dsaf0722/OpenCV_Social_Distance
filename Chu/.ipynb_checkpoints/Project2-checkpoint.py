import math
import cv2
import caffe
import numpy as np
import imutils
from imutils.video import VideoStream
import time

# esc to shut dowm
# frame_size = (640*480)


def gen_bbox(hotmap, offset, scale, th):
	h, w = hotmap.shape
	stride = 2
	win_size = 12
	hotmap = hotmap.reshape((h, w))
	keep = hotmap > th
	pos = np.where(keep)
	score = hotmap[keep]
	offset = offset[:, keep]
	x, y = pos[1], pos[0]
	x1 = stride * x
	y1 = stride * y
	x2 = x1 + win_size
	y2 = y1 + win_size
	x1 = x1 / scale
	y1 = y1 / scale
	x2 = x2 / scale
	y2 = y2 / scale
	bbox = np.vstack([x1, y1, x2, y2, score, offset]).transpose()
	return bbox.astype(np.float32)

def nms(dets, thresh, meth='Union'):
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		if meth == 'Union':
			ovr = inter / (areas[i] + areas[order[1:]] - inter)
		else:
			ovr = inter / np.minimum(areas[i], areas[order[1:]])
		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]
	return keep

def bbox_reg(bboxes):
	w = bboxes[:, 2] - bboxes[:, 0]
	h = bboxes[:, 3] - bboxes[:, 1]
	bboxes[:, 0] += bboxes[:, 5] * w
	bboxes[:, 1] += bboxes[:, 6] * h
	bboxes[:, 2] += bboxes[:, 7] * w
	bboxes[:, 3] += bboxes[:, 8] * h
	return bboxes

def make_square(bboxes):
	x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2
	y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2
	w = bboxes[:, 2] - bboxes[:, 0]
	h = bboxes[:, 3] - bboxes[:, 1]
	size = np.vstack([w, h]).max(axis=0).transpose()
	bboxes[:, 0] = x_center - size / 2
	bboxes[:, 2] = x_center + size / 2
	bboxes[:, 1] = y_center - size / 2
	bboxes[:, 3] = y_center + size / 2
	return bboxes

def crop_face(img, bbox, wrap=True):
	height, width = img.shape[:-1]
	x1, y1, x2, y2 = bbox
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
		print('[WARN] ridiculous x1, y1, x2, y2')
		return None
	if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
		# out of boundary, still crop the face
		if not wrap:
			return None
		h, w = y2 - y1, x2 - x1
		patch = np.zeros((h, w, 3), dtype=np.uint8)
		vx1 = 0 if x1 < 0 else x1
		vy1 = 0 if y1 < 0 else y1
		vx2 = width if x2 > width else x2
		vy2 = height if y2 > height else y2
		sx = -x1 if x1 < 0 else 0
		sy = -y1 if y1 < 0 else 0
		vw = vx2 - vx1
		vh = vy2 - vy1
		patch[sy:sy+vh, sx:sx+vw] = img[vy1:vy2, vx1:vx2]
		return patch
	return img[y1:y2, x1:x2]

def mtcnn_detection(img, scales, width, height):
	### pnet ###
	bboxes_in_all_scales = np.zeros((0, 4 + 1 + 4), dtype=np.float32)
	for scale in scales:
		w, h = int(math.ceil(scale * width)), int(math.ceil(scale * height))
		data = cv2.resize(img, (w, h))
		data = data.transpose((2, 0, 1)).astype(np.float32) # order now: ch, height, width
		data = (data - 128) / 128
		data = data.reshape((1, 3, h, w)) # order now: batch, ch, height, width
		pnet.blobs['data'].reshape(*data.shape)
		pnet.blobs['data'].data[...] = data
		pnet.forward()
		prob = pnet.blobs['prob'].data
		bbox_pred = pnet.blobs['bbox_pred'].data
		bboxes = gen_bbox(prob[0][1], bbox_pred[0], scale, 0.6)
		keep = nms(bboxes, 0.5) # nms in each scale
		bboxes = bboxes[keep]
		bboxes_in_all_scales = np.vstack([bboxes_in_all_scales, bboxes])
	# nms in total
	keep = nms(bboxes_in_all_scales, 0.7)
	bboxes_in_all_scales = bboxes_in_all_scales[keep]
	bboxes_in_all_scales = bbox_reg(bboxes_in_all_scales)
	bboxes_in_all_scales = make_square(bboxes_in_all_scales)
	if len(bboxes_in_all_scales) == 0:
		return bboxes_in_all_scales


	### rnet ###
	n = len(bboxes_in_all_scales)
	data = np.zeros((n, 3, 24, 24), dtype=np.float32)
	for i, bbox in enumerate(bboxes_in_all_scales):
		face = crop_face(img, bbox[:4])
		data[i] = cv2.resize(face, (24, 24)).transpose((2, 0, 1))
	data = (data - 128) / 128
	rnet.blobs['data'].reshape(*data.shape)
	rnet.blobs['data'].data[...] = data
	rnet.forward()
	prob = rnet.blobs['prob'].data
	bbox_pred = rnet.blobs['bbox_pred'].data
	prob = prob.reshape(n, 2)
	bbox_pred = bbox_pred.reshape(n, 4)
	keep = prob[:, 1] > 0.7
	bboxes_in_all_scales = bboxes_in_all_scales[keep]
	bboxes_in_all_scales[:, 4] = prob[keep, 1]
	bboxes_in_all_scales[:, 5:9] = bbox_pred[keep]
	keep = nms(bboxes_in_all_scales, 0.7)
	bboxes_in_all_scales = bboxes_in_all_scales[keep]
	bboxes_in_all_scales = bbox_reg(bboxes_in_all_scales)
	bboxes_in_all_scales = make_square(bboxes_in_all_scales)
	if len(bboxes_in_all_scales) == 0:
		return bboxes_in_all_scales

	### onet ###
	n = len(bboxes_in_all_scales)
	data = np.zeros((n, 3, 48, 48), dtype=np.float32)
	for i, bbox in enumerate(bboxes_in_all_scales):
		face = crop_face(img, bbox[:4])
		data[i] = cv2.resize(face, (48, 48)).transpose((2, 0, 1))
	data = (data - 128) / 128
	onet.blobs['data'].reshape(*data.shape)
	onet.blobs['data'].data[...] = data
	onet.forward()
	prob = onet.blobs['prob'].data
	bbox_pred = onet.blobs['bbox_pred'].data
	prob = prob.reshape(n, 2)
	bbox_pred = bbox_pred.reshape(n, 4)
	keep = prob[:, 1] > 0.4
	bboxes_in_all_scales = bboxes_in_all_scales[keep]
	bboxes_in_all_scales[:, 4] = prob[keep, 1]
	bboxes_in_all_scales[:, 5:9] = bbox_pred[keep]
	bboxes_in_all_scales = bbox_reg(bboxes_in_all_scales)
	keep = nms(bboxes_in_all_scales, 0.5, 'Min')
	bboxes_in_all_scales = bboxes_in_all_scales[keep]
	return bboxes_in_all_scales

# pnet = caffe.Net('proto/p.prototxt', 'tmp/pnet_iter_327000.caffemodel', caffe.TEST)
# rnet = caffe.Net('proto/r.prototxt', 'tmp/rnet_iter_91000.caffemodel', caffe.TEST)
# onet = caffe.Net('proto/o.prototxt', 'tmp/onet_iter_32000.caffemodel', caffe.TEST)

pnet = caffe.Net('mtcnn-head-detection/proto/p.prototxt', 'mtcnn-head-detection/tmp/pnet_iter_446000.caffemodel', caffe.TEST)
rnet = caffe.Net('mtcnn-head-detection/proto/r.prototxt', 'mtcnn-head-detection/tmp/rnet_iter_116000.caffemodel', caffe.TEST)
onet = caffe.Net('mtcnn-head-detection/proto/o.prototxt', 'mtcnn-head-detection/tmp/onet_iter_90000.caffemodel', caffe.TEST)

def find_head_width(switch):
    if switch == 1:
        head_width = [np.abs(list_[0][0]-list_[0][2])]
        return head_width

# find distance to camera  
focalLength = 1023 # U11_cam
# head_real_width average = 13.9cm = 5.47 inches
KNOWN_WIDTH = 13.9

def distance_to_camera(knownWidth, focalLength, perWidth):  
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth 
        

# def head_detection(switch):
#     if switch == 1:      
cap = cv2.VideoCapture(0)
ret, img = cap.read()
min_size = 24
factor = 0.709
base = 12. / min_size     
height, width = img.shape[:-1]
l = min(width, height)
l *= base
scales = []
while l > 12:
    scales.append(base)
    base *= factor
    l *= factor

while(True):
    k = cv2.waitKey(1) & 0xff
    ret, img = cap.read()
    boxes_list = []
    onet_boxes = mtcnn_detection(img, scales, width, height)
    imgdraw_onet = img.copy() 

    list_ = []
    for i in range(len(onet_boxes)):
        time.sleep(0.001)
        x1, y1, x2, y2, score = onet_boxes[i, :5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  
        # when box_center_x = 400(center of frame) 
        box_width_before_standardize = np.abs(x1-x2)
        box_center_x,box_center_y = [int((x1+x2)/2),int((y1+y2)/2)]
        # frame_center = (320,240)
        pix_boxcenter_to_framecenter_x = np.abs(int((x1+x2)/2-320))                
        # if box_center_x = 320, box_width_before_standardize = 60 and 
        # if box_end_x = 640, shrink_box_width = 80
        # so find the box_width after box_center_x change and reply it
        # >>>  shrink_box_width = 1/3*box_width_before_standardize*boxcenter_to_framecenter/320+box_width
        # >>>  box_width = shrink_box_width - 1/3*box_width_before_standardize*boxcenter_to_framecenter/320
        biggest_shrink = 1/3 * box_width_before_standardize
        box_width = box_width_before_standardize - biggest_shrink * pix_boxcenter_to_framecenter_x/320
#                 print([box_center_x,box_width_before_standardize,box_width])
#                 print(x1,x2,y1,y2,box_center_x)
        shrink = KNOWN_WIDTH/box_width 
        real_boxcenter_to_framecenter_x = pix_boxcenter_to_framecenter_x * shrink
        dis_to_cam_cm = distance_to_camera(KNOWN_WIDTH, focalLength,box_width)
#                 print(dis_to_cam_cm,box_width)
        rectangle_color = (255,0,0)
        list_.append([x1, y1, x2, y2,box_width,shrink,box_center_x,box_center_y,rectangle_color,pix_boxcenter_to_framecenter_x,real_boxcenter_to_framecenter_x,dis_to_cam_cm])
        boxes = len(list_)
        h_distance = 0
        h_distance_to_others = []
        real_distance_of_objects = 0
        real_distance_of_objects_list = []
        cv2.rectangle(imgdraw_onet, (x1, y1), (x2, y2), (255,0,0), 2) 
        if boxes >= 2 and score > 0.99:
            for j in range(boxes):
                for n in range(boxes):
                    if j != n and n > j:
                        # list_[j][6] = box_center_x
                        h_distance = np.abs(list_[j][6]-list_[n][6])
                        h_distance_to_others.append([boxes,j,n,h_distance])
                        # list_[j][-2] = list_[n][-2] = real_boxcenter_to_framecenter_x
                        if list_[j][-2] == 0 and list_[n][-2] != 0:
                            A1 = 90
                            A2 = 90 - np.abs(math.acos(list_[n][-2]/list_[n][-1])*180/math.pi)
                        elif list_[n][-2] == 0 and list_[j][-2] != 0:
                            A1 = 90 - np.abs(math.acos(list_[j][-2]/list_[j][-1])*180/math.pi) 
                            A2 = 90
                        elif list_[j][-2] != 0 and list_[n][-2] != 0:
                            A1 = 90 - np.abs(math.acos(list_[j][-2]/list_[j][-1])*180/math.pi) 
                            A2 = 90 - np.abs(math.acos(list_[n][-2]/list_[n][-1])*180/math.pi)
                        # list[j][6] = box_center_x
                        if list_[j][6] < 320 and list_[n][6] < 320:
                            A = np.abs(A1-A2)
                        elif list_[j][6] > 320 and list_[n][6] > 320:
                            A = np.abs(A1-A2)
                        else:
                            A = A1 + A2
                        A_ = A * math.pi / 180  
                        # to show 0.2f use round(,2)
                        # list_[j][-1] = dis_to_cam_cm
                        real_distance_of_objects = round(np.abs((list_[n][-1])**2 + (list_[j][-1])**2 - (2*(list_[n][-1])*(list_[j][-1]) * math.cos(A_)))**0.5,2)
#                     print(h_distance_to_others)
                        real_distance_of_objects_list.append([boxes,j,n,real_distance_of_objects])
#                                 print(real_distance_of_objects_list)
                        if len(real_distance_of_objects_list) >= 1:
                            for result in real_distance_of_objects_list: 
#                                         time.sleep(0.01)
                                start_x = list_[result[1]][6] if list_[result[1]][6] < list_[result[2]][6] else list_[result[2]][6]
                                start_y = list_[result[1]][7] if list_[result[1]][6] < list_[result[2]][6] else list_[result[2]][7]
                                end_x = list_[result[2]][6] if list_[result[2]][6] > list_[result[1]][6] else list_[result[1]][6]
                                end_y = list_[result[2]][7] if list_[result[2]][6] > list_[result[1]][6] else list_[result[1]][7]
                                center_of_objects_x, center_of_objects_y =  int((start_x + end_x)/2), int((start_y + end_y)/2)
                                if result[-1] < 150:
                                    list_[result[1]][-4] = (0,0,255) 
                                    list_[result[2]][-4] = (0,0,255)
#                                             print(result)
                                    cv2.rectangle(imgdraw_onet, (list_[result[1]][0],list_[result[1]][1]), (list_[result[1]][2], list_[result[1]][3]), list_[result[1]][-4], 2) 
                                    cv2.rectangle(imgdraw_onet, (list_[result[2]][0], list_[result[2]][1]), (list_[result[2]][2], list_[result[2]][3]), list_[result[2]][-4], 2)
                                    # show center_line_of_objects
#                                             cv2.line(imgdraw_onet,(list_[result[1]][6],list_[result[1]][7]),(list_[result[2]][6],list_[result[2]][7]),list_[result[1]][-4],2)
#                                             cv2.line(imgdraw_onet,(list_[result[1]][6],list_[result[1]][7]),(list_[result[2]][6],list_[result[2]][7]),(0,255,0),2)
                                    # show dsitance_cm_of_objects
#                                             cv2.putText(imgdraw_onet,'%0.2fcm'%result[-1],(center_of_objects_x-50,center_of_objects_y-10),cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
#                                     cv2.putText(imgdraw_onet,f'person {result[1]}',(list_[result[1]][0],list_[result[1]][1]-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
#                                     cv2.putText(imgdraw_onet,f'person {result[2]}',(list_[result[2]][0],list_[result[2]][1]-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                                else:
                                    cv2.rectangle(imgdraw_onet, (list_[result[1]][0], list_[result[1]][1]), (list_[result[1]][2], list_[result[1]][3]), list_[result[1]][-4], 2) 
                                    cv2.rectangle(imgdraw_onet, (list_[result[2]][0], list_[result[2]][1]), (list_[result[2]][2], list_[result[2]][3]), list_[result[2]][-4], 2) 
#                                             cv2.line(imgdraw_onet,(list_[result[1]][6],list_[result[1]][7]),(list_[result[2]][6],list_[result[2]][7]),list_[result[1]][-4],2)
#                                             cv2.line(imgdraw_onet,(list_[result[1]][6],list_[result[1]][7]),(list_[result[2]][6],list_[result[2]][7]),(0,255,0),2)
#                                             cv2.putText(imgdraw_onet,'%0.2fcm'%result[-1],(center_of_objects_x-50,center_of_objects_y-10),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
#                                     cv2.putText(imgdraw_onet,f'person {result[1]}',(list_[result[2]][0],list_[result[2]][1]-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
#                                     cv2.putText(imgdraw_onet,f'person {result[2]}',(list_[result[2]][0],list_[result[2]][1]-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
#                                     print(result,list_[result[1]][-4],list_[result[2]][-4])

        cv2.putText(imgdraw_onet,'person',(x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv2.putText(imgdraw_onet,'%.02fcm'%list_[i][-1],(x1,y1-30),cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))

#         cv2.putText(imgdraw_onet, '%.03f'%score, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    imgdraw_onet = imutils.resize(imgdraw_onet,width=800)
    cv2.imshow("head_detection", imgdraw_onet)


    if k == 27 :
        break
cap.release()
cv2.destroyAllWindows()
#         return(list_)

