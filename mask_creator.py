import cv2
import numpy as np
import sys
from hikvisionapi import Client

"""
This script will create a mask of a video or live feed with
"""
# originalFile = '54_MF.mp4'
if len(sys.argv) < 3:
    print("Error: Input File Missing")
    exit()
flag = int(sys.argv[2])
originalFile = sys.argv[1]
if flag == 0:
    video_capture= cv2.VideoCapture(int(originalFile))
    mask_file = originalFile + '_mask.txt'
    rectangles_file = originalFile + '_rectangles.txt'
elif flag == 1:
    video_capture= cv2.VideoCapture(originalFile)
    mask_file = originalFile.replace('.mp4', '_mask.txt')
    rectangles_file = originalFile.replace('.mp4', '_rectangles.txt')
elif flag == 2:
    video_capture = cv2.VideoCapture()
    video_capture.open(f"rtsp://admin:aa123456@{originalFile}:554/Streaming/channels/101/")
    mask_file = originalFile + '_mask.txt'
    rectangles_file = originalFile + '_rectangles.txt'

# Check if the file was opened successfully
ret, current_frame = video_capture.read()
previous_frame = current_frame
if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the video's frame width and height
frame_width = int(video_capture.get(3)) // 5 * 2
frame_height = int(video_capture.get(4)) // 5 * 2

'''
# Define the codec and create a VideoWriter object for output
out_name = originalFile.replace('.mp4', '_output.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
out = cv2.VideoWriter(out_name, fourcc, 30, (frame_width, frame_height), isColor=False)
'''
# Process frames
# Callback function to get mouse position
pos = np.zeros((4 * 10 + 1, 2))
g_i = 0
g_total = 0
def get_mouse_position(event, x, y, flag, param):
    """
    Gets the x and y coordinates of the left mouse button position on 
        the window when clicked, and saves them in an array, additionally
        when the number of clicks is a multiple of 4 we concatenate g_i so
        in the future we know the number of rectangles present
    Args:
        event (_type_): cv2 event, we only care about cv2.EVENT_LBUTTONDOWN
        x (int): coordinate of the mouse position
        y (int): coordinate of the mouse position
        flag (_type_): necessary function parameters for setMouseCallback
        param (_type_): necessary function parameters for setMouseCallback
    """
    global g_i
    global g_total
    # Check if left mouse button is clicked and gets 4 clicks max
    if event == cv2.EVENT_LBUTTONDOWN:
        pos[g_i][0] = x
        pos[g_i][1] = y
        print("Click (x={}, y={})".format(pos[g_i][0],  pos[g_i][1]))
        g_i += 1
        if (g_i % 4 == 0 and g_i != 0):
            g_total += 1
pos = np.asarray(pos, dtype = 'int')

cv2.namedWindow('frame diff', cv2.WND_PROP_FULLSCREEN)
cv2.setMouseCallback('frame diff', get_mouse_position)

counter = 0
current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
previous_frame_gray = None

def draw_rectangle_point_by_point(frame, pos, t):
    """
    Using cv2.line we connect 4 points (group in intervals of 4 decided by t)
        with lines in the frame
    Args:
        frame (numpy.ndarray): cv2 video frame
        pos (int*): array containing all the rectangles points
        t (int): variable indicating the points belonging to the same rectangle
    """
    color = (0, 255, 0)  # Green color (BGR format)
    thick = 2
    i = 4 * t
    cv2.line(frame, (pos[0 + i][0], pos[0 + i][1]), (pos[1 + i][0], pos[1 + i][1]), color, thick)
    cv2.line(frame, (pos[1 + i][0], pos[1 + i][1]), (pos[2 + i][0], pos[2 + i][1]), color, thick)
    cv2.line(frame, (pos[2 + i][0], pos[2 + i][1]), (pos[3 + i][0], pos[3 + i][1]), color, thick)
    cv2.line(frame, (pos[3 + i][0], pos[3 + i][1]), (pos[0 + i][0], pos[0 + i][1]), color, thick)


while (video_capture.isOpened()):
    ret, current_frame = video_capture.read()
    t = 0

    if not ret:
        break

    if counter == 0 or counter % 15 == 0:
        current_frame_gray = current_frame
        current_frame_gray = cv2.resize(current_frame_gray, (frame_width, frame_height))

        if previous_frame_gray is not None:
            frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
        previous_frame_gray = current_frame_gray

    while (t < g_total):
        if (g_i >= 4):
            draw_rectangle_point_by_point(current_frame_gray, pos, t)
        t += 1

    if counter >= 2 * 15:
        #out.write(frame_diff)
        cv2.imshow('frame diff', frame_diff)
    key = cv2.waitKey(10)
    if key == 27:  # 27 = ESC
        break
    elif key & 0xFF == ord('r'): #reset the video, points clicked and variables
        video_capture = cv2.VideoCapture(originalFile)
        previous_frame_gray = None
        counter = 0
        pos = np.zeros((4 * 10 + 1, 2))
        pos = np.asarray(pos, dtype = 'int')
        g_i = 0
        g_total = 0
    counter += 1

# save rectangle corners to txt file to immediately load them next run
np.savetxt(rectangles_file, pos)
# Create a black image (all zeros) of the same size as your camera feed frame
mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
rectangle_corner = []
index = 0
while (g_total > 0):
        g_total -= 1
        rectangle_corner.append(pos[4 * g_total: 4 + 4 * g_total])
        cv2.fillConvexPoly(mask, rectangle_corner[index], 255)
        index += 1
print(index)
np.savetxt(mask_file, mask, fmt='%d')
# Release video objects
video_capture.release()
#out.release()
# Close all OpenCV windows
cv2.destroyAllWindows()