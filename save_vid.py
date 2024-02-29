import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import concurrent.futures
import socket
import os
import csv
from datetime import datetime
from autodistill_yolov8 import YOLOv8

#https://www.onlogic.com/company/io-hub/how-to-ssh-into-raspberry-pi/
#https://stackoverflow.com/questions/66410168/can-i-send-python-script-outputs-to-a-raspberry-pi-over-ssh-wifi

def jam_detector(originalFile, flag):
    mask_file = originalFile.replace('.mp4', '') + '_mask.txt'
    if flag == 1:
        video = cv2.VideoCapture(originalFile)

    ret, current_frame = video.read()
    if not video.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # Load the mask from a file
    mask = np.loadtxt(mask_file)
    mask = np.asarray(mask, dtype = np.uint8)
    frame_counter = 0
    previous_frame_gray = None
    frame_interval = 15

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    output_video = cv2.VideoWriter('test.mp4', fourcc, video.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    while (video.isOpened()):
        ret, current_frame = video.read()
        if not ret:
            break
        if frame_counter == 0 or frame_counter % frame_interval == 0:
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            #current_frame_gray = cv2.resize(current_frame_gray, (frame_width, frame_height))
            if previous_frame_gray is not None:
                frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
                #frame_diff = cv2.bitwise_and(frame_diff, frame_diff, mask=mask)
                output_video.write(frame_diff)
            previous_frame_gray = current_frame_gray

        if frame_counter >= 2 * frame_interval:
            cv2.imshow('frame diff', frame_diff)
            #cv2.imshow('frame diff', current_frame)
        if cv2.waitKey(15) == 27:  # 27 = ESC
            break
        frame_counter += 1
    # Release video objects and close all windows
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

if len(sys.argv) == 3:
    w = sys.argv[1]
    jam_detector(w, int(sys.argv[2]))
