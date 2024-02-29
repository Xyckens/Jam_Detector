import cv2
import numpy as np
import socket
import os
import csv
from datetime import datetime
from autodistill_yolov8 import YOLOv8
from threading import Thread

#https://www.onlogic.com/company/io-hub/how-to-ssh-into-raspberry-pi/
#https://stackoverflow.com/questions/66410168/can-i-send-python-script-outputs-to-a-raspberry-pi-over-ssh-wifi
ip = "10.16.40.89"

def rectangle_func(file_path):
    """
    Iterates through file_path and stores the mask (rectangles) value points, then gets
        the max and min value for x and y (the extremes) and returns them
    Args:
        file_path (string): path to txt file containing the rectangles' points

    Returns:
        (int*): max and min value for x and y of all the rectangles
    """
    l = 0
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(' '))
            if (x == 0.0 and y == 0.0):
                break
            l += 1
    rectangles = np.zeros((l, 2))
    l = 0
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(' '))
            if (x == 0.0 and y == 0.0):
                break
            rectangles[l][0], rectangles[l][1] = x, y
            l += 1
    print(rectangles)
    print('max x', np.max(rectangles[:,0]))
    print('min x', np.min(rectangles[:,0]))
    print('max y', np.max(rectangles[:,1]))
    print('min y', np.min(rectangles[:,1]))
    extremes = [np.max(rectangles[:,0]), np.min(rectangles[:,0]), np.max(rectangles[:,1]), np.min(rectangles[:,1])]
    extremes = np.asarray(extremes, dtype = int)
    return (extremes)

# Movement Checker
def movement_checker(frame, extremes):
    """
    For each pixel of frame between the extremes values, depending on their value
        we seperathem between movmnt or not and count them to calculate the 
        percentage of pixels above an arbitary value
    Args:
        frame (numpy.ndarray): cv2 video frame/ image
        extremes (int array): max and min value for x and y that gives us the 
                            boundaries of points of interest for frame
    Returns:
        (float): The percentage of pixels above a set threshold value
    """
    movmt = 0
    movmtless = 0
    avy = 0
    for y  in range (extremes[3], extremes[2]):
        avx = 0
        for x  in range (extremes[1], extremes[0]):
            if (frame[y][x] > 5):
                movmt += 1
                avx += frame[y][x]
            elif (frame[y][x] > 0 and frame[y][x] <= 5):
                avx += frame[y][x]
                movmtless += 1
            elif (frame[y][x] == 0):
                x += 1
            x += 1
        avy += avx / (extremes[0] - extremes[1])
        y += 1
    if movmt == movmtless == 0:
        print("Error: Video isn't loading correctly")
        exit()
    percentage = ((movmt / (movmt + movmtless)) * 100)
    print(f'movement percentage {percentage:.1f}%' ,\
        f'with avg pixel strength {(avy / (extremes[2] - extremes[3])):.1f}')
    return percentage

def save_jam(frame, name, counter):
    """
    Saves the frame of a detected jam in a folder with it's name and the date when it was detected
    Args:
        frame (numpy.ndarray): cv2 video frame/ image
        name (str): file name
    """
    if counter > 11:
        return
    parent_dir = "C:\\Users\\fvieira\\OneDrive - Baglass\\Desktop\\mp4" 
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M")
    path = os.path.join(parent_dir, name.replace('.mp4',''))
    if not os.path.exists(path):
        os.mkdir(path)
    frame_filename = os.path.join(path, f'{formatted_time}.jpg')
    cv2.imwrite(frame_filename, frame)

def bottle_detector(frame, movmt, counter, client, model):
    """
    We analize the frame for bottles, depending if we detect them we change the
        message acordingly, additionally, when we detect bottles and still a 
        movement above 30% we assume this means the jam is appening right now 
        where we can see it. Finally we send it to the client where we'll raise the alarm
    Args:
        frame (numpy.ndarray): cv2 video frame/ image
        movmt (float): percentage of movement calculated in previous functions
        counter (int): number of times there was detected a low percentage of movmt
        client (socket.socket): variable to be abble to communicate with raspberry pi
        model (autodistill_yolov8.yolov8.YOLOv8): custom autodistill model built from yolov8
            and trained to detect glass containers
    """
    results = model.predict(frame)
    #cv2.imshow('Detection', frame)
    ar = results[0].boxes.xyxy #nbr of detections
    if (counter > 11):
        return
    if len(ar) == 0:
        msg = "\nBEFORE"
    elif (movmt > 30):
        msg = "\nHERE"
    else:
        msg = "\nFURTHER DOWN"
    #client.send(msg.encode())

def append_to_csv(file_path, mov_persistance, low_percnt, avg):
    # Open the CSV file in append mode
    current_time = datetime.now()
    formatted_time = current_time.strftime("%YY%mM%dD_%Hh%Mm")
    with open(file_path, 'a', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        row = [mov_persistance] + [low_percnt] + [avg] + [formatted_time]
        # Write the data to the CSV file
        csv_writer.writerow(row)

def jam_detector(originalFile, flag):
    """
    Upon recieving a valid file path, will open the needed txt files (mask and rectangles)
        and start the video capture (also resizing it), during the video iteration we 
        compare the current frame with a previous one (15 frames ago) and apply the mask
        to send it to the movement_checker function that returns a percentage, if that value
        is to low we start counting till we reach 10 low percentages in a row which it's been
        considered to qualify as a jam, so we send the data to bottle_detector, meanwhile
        we store as much information as we can to return it in the end to plot it
    Args:
        originalFile (str): path to video/ live feed
    Returns:
        (array of array of ints): multiple arrays to plot the data in graphs
    """
    csv= 'csv_' +  originalFile.replace('.mp4','') + '.csv'
    mask_file = originalFile.replace('.mp4', '') + '_mask.txt'
    rectangles_file = originalFile.replace('.mp4', '') + '_rectangles.txt'
    if flag == 0:
        video = cv2.VideoCapture(int(originalFile))
    elif flag == 1:
        video = cv2.VideoCapture(originalFile)
    elif flag == 2:
        video = cv2.VideoCapture()
        video.open(f"rtsp://admin:aa123456@{originalFile}:554/Streaming/channels/101/")

    ret, current_frame = video.read()
    if not video.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    display_w = frame_width // 5 * 2 # Reduce width by a factor of 0.4
    display_h = frame_height // 5 * 2

    # Load the mask from a file
    mask = np.loadtxt(mask_file)
    mask = np.asarray(mask, dtype = np.uint8)

    # Load the rectangles from a file and save the extremes
    extremes = rectangle_func(rectangles_file)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #client.connect((ip, 8080))

    #cv2.namedWindow('frame diff', cv2.WND_PROP_FULLSCREEN)

    model = YOLOv8("content\\runs\\detect\\train\\weights\\best.pt")
    frame_counter = 0
    previous_frame_gray = None
    mov_persistance = 0
    frame_interval = 15
    mov_change_threshold = 10
    percentage = []
    frame = []
    percentage_low = []
    frame_low = []
    frame_vert = []
    time_flag = 0
    while (video.isOpened()):
        ret, current_frame = video.read()
        current_frame = cv2.resize(current_frame, (display_w, display_h))
        if not ret:
            break
        if frame_counter == 0 or frame_counter % frame_interval == 0:
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            if previous_frame_gray is not None:
                frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
                frame_diff = cv2.bitwise_and(frame_diff, frame_diff, mask=mask)
                percentage.append(movement_checker(frame_diff, extremes))
                avg = sum(percentage)/ len(percentage)
                if (avg >= percentage[-1] + mov_change_threshold):
                    mov_persistance += 1
                    print(f'{mov_persistance} low {percentage[-1]:.1f} %')
                    percentage_low.append(percentage[-1])
                    frame_low.append(frame_counter)
                    if (mov_persistance == 1):
                        frame_vert.append(frame_counter)
                    if (mov_persistance > 10):
                        print("ENCRAVOU")
                        yolo = cv2.resize(current_frame,  (display_w, display_h))
                        bottle_detector(cv2.bitwise_and(yolo, yolo, mask=mask), percentage[-1], mov_persistance, client, model)
                        save_jam(current_frame, originalFile, mov_persistance)
                    percentage.pop()
                else:
                    if (mov_persistance > 1):
                        frame_vert.append(frame_counter)
                    mov_persistance = 0
                    frame.append(frame_counter)
            previous_frame_gray = current_frame_gray

        if (datetime.now().second == 0 and time_flag == 0):
            time_flag = 1
            append_to_csv(csv, 0, 0, avg)
        elif (time_flag != 0 and datetime.now().second != 0):
            time_flag = 0
        if frame_counter >= 2 * frame_interval:
            #cv2.imshow(originalFile, frame_diff)
            cv2.imshow(originalFile, current_frame)
        if cv2.waitKey(1) == 27:  # 27 = ESC
            break
        frame_counter += 1
    # Release video objects and close all windows
    video.release()
    #client.close()
    return (frame, frame_low, frame_vert, percentage, percentage_low)

threads = []
#sources = [['0', 'web_cam'], ['phone.mp4', 'video'], ['172.17.4.131', 'live'], ['51_MF.mp4', 'video']]
sources = [['cam_02.mp4', 1], ['cam_14.mp4', 1], ['cam_29.mp4', 1], ['cam_b4.mp4', 1]]
for i in range(len(sources)):
    thread = Thread(target=jam_detector, args=(sources[i][0], sources[i][1]))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()
cv2.destroyAllWindows()
