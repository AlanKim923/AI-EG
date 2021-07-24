import cv2
import numpy as np
from utils.opv import OpvModel  
import matplotlib.pyplot as plt
import serial
import time

def switch():
    global ser
    ser.write('a'.encode())
    print('switch')

def DrawBoundingBoxes(predictions, image, conf = 0.6): # 예측과 이미지는 기본 코드에서 나옵니다.
    global Detected
    global offCnt
    global onCnt
    global DWtime
    global DWtimeFrame

    canvas = image.copy()                             # 원본 이미지를 수정하는 대신 복사
    predictions = predictions[0][0]
    confidence = predictions[:,2]
    topresults = predictions[(confidence>conf)]
    (h,w) = canvas.shape[:2]
    for detection in topresults:
        box = detection[3:7] * np.array([w, h, w, h])
        (xmin, ymin, xmax, ymax) = box.astype("int")

        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
        cv2.putText(canvas, str(round(detection[2]*100,1))+"%", (xmin, ymin),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)
    cv2.putText(canvas, str(len(topresults))+" persons(s) detected", (50,50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)

    if time.time() > DWtime:
        if len(topresults) > 0:
            if (not Detected) and time.time() > DWtimeFrame:
                if onCnt >= 1:
                    Detected = True
                    onCnt = 0
                    DWtimeFrame = time.time() + 4
                    print('on')
                    switch()
                else:
                    onCnt += 1
                    offCnt = 0
        else:
            if Detected and time.time() > DWtimeFrame:
                if offCnt >= 5:
                    Detected = False
                    offCnt = 0
                    DWtimeFrame = time.time() + 4
                    print('off')
                    switch()
                else:
                    offCnt += 1
                    onCnt = 0
        DWtime = time.time() + 0.5
        
    return canvas


mymodel2 = OpvModel("person-detection-retail-0013",device="CPU", fp="FP32")


windowName = "EG"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
camera = cv2.VideoCapture(0) #'첫" 번째'카메라 (웹캠)로 VideoCapture 개체 만들기

Detected = False
offCnt = 0;
onCnt = 0;
DWtime = time.time()
DWtimeFrame = time.time()

ser = serial.Serial(port='/dev/ttyUSB0', baudrate=9600)

while(True):
    ret, frame = camera.read()             # 프레임별로 캡처    
    predictions = mymodel2.Predict(frame)

    frame = cv2.resize(frame, dsize=(1280, 720), interpolation=cv2.INTER_AREA)

    cv2.imshow(windowName,DrawBoundingBoxes(predictions,frame))
    if cv2.waitKey(1) & 0xFF == ord(' '):  # 스페이스 바가 감지되면 중지
        break

camera.release()                           # 스페이스 바가 감지 된 후 camera 객체를 해제
cv2.destroyAllWindows()
ser.close()
