import cv2
import numpy as np
import time
from time import strftime
from flask import Flask, render_template, Response
import imutils

app = Flask(__name__)

load_from_disk = True
if load_from_disk:
#     blue = np.load('./color/blue.npy') # 偵測藍色物件
#     red = np.load('./color/red.npy') # 偵測紅色物件
    yellow = np.load('./op/color/yellow.npy') # 偵測黃色物件

# 畫筆初始顏色粗細


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    pen_color = (255,0,0)
    thickness = 5
    
    cap = cv2.VideoCapture(0)

    # 左上角切換功能圖片 resize
    pen_img = cv2.resize(cv2.imread('./op/img/pen.png',1), (60, 60))
    eraser_img = cv2.resize(cv2.imread('./op/img/eraser.png',1), (60, 60))

    kernel = np.ones((5,5),np.uint8)

#     cv2.namedWindow('virtual pen', cv2.WINDOW_NORMAL)

    # canvas 為畫布
    canvas = None

    # 使用Background Subtractor MOG 2背景分離指令(不偵測陰影)
    backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

    # This threshold determines the amount of disruption in the background.
    # 設定背景干擾的門檻值
    background_threshold = 600

    # 告訴你正在使用比或橡皮擦的變數
    switch = 'Pen'

    # 此變數讓我們可以監控上一次切換所間隔的時間
    last_switch = time.time()

    # 初始化x1,y1座標
    x1,y1=0,0

    # Threshold for noise
    # 干擾閾值
    noiseth = 800

    # Threshold for wiper, the size of the contour must be bigger than this for us to clear the canvas
    # 清除畫布的閾值,輪廓大小如超過此值就會清除畫布
    wiper_thresh = 40000

    # 告訴我們何時清除畫布的變數
    clear = False

    while(1):

        _, frame = cap.read()
        frame = cv2.flip( frame, 1 )

        cv2.putText(frame,'<- touch',(65,30),cv2.FONT_HERSHEY_DUPLEX, 0.6, (200,100,255), 1, cv2.LINE_AA)

        # 將畫布初始化為全黑圖片
        if canvas is None:
            canvas = np.zeros_like(frame)

        # 取框架的左上角並應用背景分離
        top_left = frame[0: 50, 0: 50]
        fgmask = backgroundobject.apply(top_left)

        # Note the number of pixels that are white, this is the level of disruption.
        switch_thresh = np.sum(fgmask==255)

        # 如果干擾大於背景閾值，且離上一次切換之後已有段時間，則可切換功能
        if switch_thresh>background_threshold and (time.time()-last_switch) > 1:

            # Save the time of the switch. 
            last_switch = time.time()

            if switch == 'Pen':
                switch = 'Eraser'
            else:
                switch = 'Pen'


        # BGR轉HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定義顏色範圍
        if load_from_disk:
    #         lower_range = blue[0]
    #         upper_range = blue[1]
    #         lower_range = red[0]
    #         upper_range = red[1]
            lower_range = yellow[0]
            upper_range = yellow[1]


        # 從HSV圖像中擷取藍色，黃色即獲得相應的遮罩
        # cv2.inRange()函數則是只顯示遮罩範圍內的顏色
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # 執行形態學運算以消除雜訊
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 2)

        # 找到物件邊緣
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)

        # 確保輪廓存在，並且輪廓的大小大於干擾閾值。
        if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:

            c = max(contours, key = cv2.contourArea)    
            x2,y2,w,h = cv2.boundingRect(c)

            # 取得輪廓範圍
            area = cv2.contourArea(c)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(frame, (int(x), int(y)), int(radius), (120, 200, 255), 2)

            # 如果沒有先前的座標點，則將偵測到的x2,y2坐標另存為x1,y1。
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2

            else:

                if switch == 'Pen':
                    # 在畫布上畫線
                    canvas = cv2.line(canvas, (x1,y1), (x2,y2), pen_color, thickness)

                else:
                    # 橡皮擦大小
                    cv2.circle(canvas, (x2, y2), 30,
                    (0,0,0), -1)


            # After the line is drawn the new points become the previous points.
            x1,y1 = x2,y2

            # Now if the area is greater than the wiper threshold then set the 
            # clear variable to True

            if area > wiper_thresh:
                cv2.putText(canvas,'Reset Screen!',(90,250), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (50,160,255), 3, cv2.LINE_AA)
                clear = True 

        else:
            # If there were no contours detected then make x1,y1 = 0
            x1,y1 = 0,0


        # 下面五行是為了使繪圖更加平順
        _ , mask = cv2.threshold(cv2.cvtColor (canvas, cv2.COLOR_BGR2GRAY), 20, 
        255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
        background = cv2.bitwise_and(frame, frame,
        mask = cv2.bitwise_not(mask))
        frame = cv2.add(foreground,background)

        # 依照現在使用的功能切換左上角圖示
        if switch != 'Pen':
            cv2.circle(frame, (x1, y1), 30, (255,255,255), -1)
            frame[0: 60, 0: 60] = eraser_img
        else:
            frame[0: 60, 0: 60] = pen_img

        frame2 = imutils.resize(frame,width=200,height=200)
        cv2.imshow('virtual pen',frame2)
        
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        k = cv2.waitKey(5) & 0xFF
        # 按Esc結束程式
        if k == 27:
            break

        # 按s鍵將所繪線條存成.png檔
        elif k == ord('s'):
            cv2.imwrite("./op/img/img"+ strftime("%m%d_%H%M%S")+ ".png", canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])

        elif k == ord('x'):
            pen_color = (0,0,255) # 紅色
            thickness = 5

        elif k == ord('c'):
            pen_color = (0,255,0) # 綠色
            thickness = 5

        elif k == ord('v'):
            pen_color = (0,255,255) # 黃色
            thickness = 5

        elif k == ord('b'):
            pen_color = (255,0,0) # 藍色
            thickness = 5

        elif k == ord('z'):
            pen_color = 0 
            thickness = 1
            
        elif k == ord('q'):
            os.system('python3 op/vpen.py')
        

        # Clear the canvas after 1 second, if the clear variable is true
        if clear == True: 
            time.sleep(1)
            canvas = None

            # And then set clear to false
            clear = False   

    cv2.destroyAllWindows()
    cap.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)