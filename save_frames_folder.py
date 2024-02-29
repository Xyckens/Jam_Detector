import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import psutil
import time
# from ultralytics import yolo

if len(sys.argv) < 2:
    print("Error: Input File Missing")
    exit()
originalFile = sys.argv[1]
video_capture = cv2.VideoCapture(originalFile)

# Check if the file was opened successfully
ret, current_frame = video_capture.read()
previous_frame = current_frame
if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

# if vid open correctly: create folder to store every frame
# Parent Directory path
parent_dir = "C:\\Users\\fvieira\\OneDrive - Baglass\\Desktop\\mp4" 
# Path
path = os.path.join(parent_dir, originalFile[:-4])
if not os.path.exists(path):
    os.mkdir(path)

# Get the video's frame width and height
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))


# Process frames

frame_count = 0
# Event handler to close the window

while (video_capture.isOpened()):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
    if frame_count % 4 == 0 and frame_count > 190:
        frame_filename = os.path.join(path, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, current_frame)
        print(frame_count)
    previous_frame = current_frame.copy()
    ret, current_frame = video_capture.read()
    if not ret:
        break
    frame_count += 1

# Release video objects
video_capture.release()
# Close all OpenCV windows
cv2.destroyAllWindows()