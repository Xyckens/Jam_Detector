import numpy as np
import cv2
from hikvisionapi import Client

cap = cv2.VideoCapture()
cap.open("rtsp://admin:aa123456@172.17.4.131:554/Streaming/channels/101/")
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# Set the window size to match the screen resolution
cv2.resizeWindow('frame', 740, 360)
while(True):
     # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    #print('test')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#https://github.com/yuriizubkov/videowall