
# draw a bounding box around the image and display it
        #box = np.int0(cv2.cv.BoxPoints(marker))
#         box = cv2.boxPoints(marker)
#         box = np.int0(box)
#         cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)


#                             if key == ord('z'):
#                                     horizontal_distance = np.abs(boxes[1][1]-boxes[0][-3]) if np.abs(boxes[1][1]-boxes[0][-3]) < np.abs(boxes[0][1]-boxes[1][-3]) else np.abs(boxes[0][1]-boxes[1][-3])                                    
#         #                             print(KNOWN_WIDTH_ALL[0][1],box_width,shrink,shrink_)
#                                     # convert pix to real_distance and convert inches to cm
#                                     real_horizontal_distance = horizontal_distance * shrink_ * 30.48 / 12
#         #                             line_start_x = boxes[0][-3] if boxes[0][-3] > boxes[0][1] else boxes[0][1]
#                                     line_start_x = boxes[0][-3] if boxes[0][-3] < boxes[1][1] else boxes[0][1]
#         #                             line_end_x = boxes[1][1] if boxes[1][1] < boxes[1][-3] else boxes[1][-3]
#                                     line_end_x = boxes[1][1] if boxes[1][1] > boxes[0][-3] else boxes[1][-3]
#                                     print(line_start_x,line_end_x)
#                                     if real_horizontal_distance > 0:
#         #                                 print(boxes[0])
#         #                                 print(boxes[0][-3],boxes[1][2])
#                                         cv2.line(frame,(line_start_x,boxes[0][-2]-150),(line_end_x,boxes[0][-2]-150),(0,0,200),2)
#                                         cv2.putText(frame,'%.2f cm' % real_horizontal_distance,
#                                     (line_start_x,boxes[0][-2]-160), cv2.FONT_HERSHEY_SIMPLEX,
#                                                  1.0,(0,0,200),2)

#                             print(f'{real_distance_of_objects} cm')
#                             y1_sin = x1_center / boxes[0][-4]
#                             y2_sin = center_x2 / boxes[1][-4]
                            
#                                 print(boxes[0][-1],boxes[1][-1])
#                                 print('%.2f cm' % real_horizontal_distance)
                            
#                             To find angle between d1 and d2 def m   
#                             d1 = boxes[0][5] if boxes[0][5] < boxes[1][5] else boxes[1][5]
#                             d2 = boxes[1][5] if boxes[1][5] > boxes[0][5] else boxes[0][5]
#                             m = np.abs((d1**2-real_horizontal_distance**2))**0.5
# #                             np.seterr(invalid='ignore')
# #                             print(d1,d2,real_horizontal_distance,m)

#                             # find cos(horizontal_distance) between d1 and m(d2)
#                             cos_horizontal_distance = (m**2+d1**2-real_horizontal_distance**2)/(2*m*d1)
#                             real_distance_between_two = (np.abs(d1**2+d2**2-2*d1*d2*cos_horizontal_distance))**0.5

 
#                             # Consider the 3D factor and use distance to camera to caculate real_distance   
                            
#                             cv2.line(frame,(obj1_center[0],obj1_center[1]-100),(obj2_center[0],obj2_center[1]-100),(200,0,0),2)
#                             cv2.putText(frame,'%.2f cm' % real_distance_between_two,
#                             (int((boxes[0][1]+boxes[0][3]/2 + boxes[1][1]+boxes[1][3]/2)/2-50),
#                              boxes[0][2]+int(boxes[0][4]/2)-100), cv2.FONT_HERSHEY_SIMPLEX,
#                                          1.0,(200,0,0),2)

                    # error method to find horizontal_distance    
#                     for count in boxes:
# #                         print(len(boxes))
#                         horizontal_distance = 0
#                         real_horizontal_distance = 0
#                         real_distance_between_two = 0
#                         real_distance_between_boxes = 0
#                         if CLASSES[idx]== KNOWN_WIDTH_ALL[-1][0] and len(boxes) >= 2:
#                             # pix_distance between two objects
#                             horizontal_distance = np.abs(boxes[1][1]-boxes[0][1])
#                             real_horizontal_distance = np.abs(inches * 30.48/ 12 * horizontal_distance / focalLength)
# #                         if real_horizontal_distance > 0:
# #                             print('%.2f cm' % real_horizontal_distance)
#                             obj1_center = (boxes[0][1]+int(boxes[0][3]/2),boxes[0][2]+int(boxes[0][4]/2))
#                             obj2_center = (boxes[1][1]+int(boxes[1][3]/2),boxes[1][2]+int(boxes[1][4]/2))
        
#                             # draw the line between two center
#                             cv2.line(frame,obj1_center,obj2_center,(0,0,200),2)
#                             # real_horizontal_distances
#                             cv2.putText(frame,'%.2f cm' % real_horizontal_distance,
#                                 (int((boxes[0][1]+boxes[0][3]/2 + boxes[1][1]+boxes[1][3]/2)/2-50),
#                                  boxes[0][2]+int(boxes[0][4]/2)-10), cv2.FONT_HERSHEY_SIMPLEX,
#                                              1.0,(0,0,200),2)

# #                             print([real_horizontal_distance,real_distance_between_two])\
                                
#                             # find boxes distance
# #                             if key == ord('z'):
# #                             box_horizontal_distance = np.abs(boxes[1][1]-boxes[0][-3]) if np.abs(boxes[1][1]-boxes[0][-3]) < np.abs(boxes[1][-2]-boxes[0][1]) else np.abs(boxes[1][-3]-boxes[0][1])
# # #                             print(box_horizontal_distance)
# #                             cos_horizontal_distance_between_box = (m**2+d1**2-box_horizontal_distance**2)/(2*m*d1)    
# #                             real_distance_between_boxes = (np.abs(d1**2+d2**2-2*d1*d2*cos_horizontal_distance_between_box))**0.5

# # #                             print(real_distance_between_boxes)
                    
#                             COLORS[idx] = (0,0,255) if real_distance_between_two < 10 else COLORS[idx] 