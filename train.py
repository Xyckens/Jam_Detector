import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import concurrent.futures
import socket
import os
from datetime import datetime
from autodistill_yolov8 import YOLOv8

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
    print(frame_filename)
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
    if flag == 0:
        video = cv2.VideoCapture(int(originalFile))
        mask_file = originalFile + '_mask.txt'
        rectangles_file = originalFile + '_rectangles.txt'
    elif flag == 1:
        video = cv2.VideoCapture(originalFile)
        mask_file = originalFile.replace('.mp4', '_mask.txt')
        rectangles_file = originalFile.replace('.mp4', '_rectangles.txt')
    elif flag == 2:
        video = cv2.VideoCapture()
        video.open(f"rtsp://admin:aa123456@{originalFile}:554/Streaming/channels/101/")
        mask_file = originalFile + '_mask.txt'
        rectangles_file = originalFile + '_rectangles.txt'

    ret, current_frame = video.read()
    if not video.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_width = int(video.get(3)) // 5 * 2
    frame_height = int(video.get(4)) // 5 * 2

    # Load the mask from a file
    mask = np.loadtxt(mask_file)
    mask = np.asarray(mask, dtype = np.uint8)

    # Load the rectangles from a file and save the extremes
    extremes = rectangle_func(rectangles_file)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #client.connect((ip, 8080))

    cv2.namedWindow('frame diff', cv2.WND_PROP_FULLSCREEN)
    model = YOLOv8("content\\runs\\detect\\train\\weights\\best.pt")
    frame_counter = 0
    previous_frame_gray = None
    p = 0
    percentage = []
    frame = []
    percentage_low = []
    frame_low = []
    frame_vert = []

    while (video.isOpened()):
        ret, current_frame = video.read()
        if not ret:
            break
        if frame_counter == 0 or frame_counter % 15 == 0:
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.resize(current_frame_gray, (frame_width, frame_height))
            if previous_frame_gray is not None:
                frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
                frame_diff = cv2.bitwise_and(frame_diff, frame_diff, mask=mask)
                percentage.append(movement_checker(frame_diff, extremes))
                if (sum(percentage)/ len(percentage) >= percentage[-1] + 10):
                    p += 1
                    print(f'{p} low {percentage[-1]:.1f} %')
                    percentage_low.append(percentage[-1])
                    frame_low.append(frame_counter)
                    if (p == 1):
                        frame_vert.append(frame_counter)
                    if (p > 10):
                        print("ENCRAVOU")
                        yolo = cv2.resize(current_frame,  (frame_width, frame_height))
                        bottle_detector(cv2.bitwise_and(yolo, yolo, mask=mask), percentage[-1], p, client, model)
                        save_jam(current_frame, originalFile, p)
                    percentage.pop()
                else:
                    if (p > 1):
                        frame_vert.append(frame_counter)
                    p = 0
                    frame.append(frame_counter)
            previous_frame_gray = current_frame_gray

        if frame_counter >= 2 * 15:
            cv2.imshow('frame diff', current_frame)
        if cv2.waitKey(1) == 27:  # 27 = ESC
            break
        frame_counter += 1
    # Release video objects and close all windows
    video.release()
    cv2.destroyAllWindows()
    #client.close()
    return (frame, frame_low, frame_vert, percentage, percentage_low)

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    Returns:
       (array of array of ints): multiple arrays with the available/working/non working ports
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports, working_ports, non_working_ports
print(len(sys.argv))
if len(sys.argv) < 3:
    a, w, n = list_ports()
    if len(w) == 0:
        print("Error: Input File Missing")
        exit()
    for i in range(len(w)):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(jam_detector, w[i], int(sys.argv[2]))
            frame, frame_low, frame_vert, percentage, percentage_low = future.result()
    #Thread(target = bottle_detector, args =(w[i]))
    i+=1
elif len(sys.argv) == 3:
    w = sys.argv[1]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(jam_detector, w, int(sys.argv[2]))
        frame, frame_low, frame_vert, percentage, percentage_low = future.result()
#https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread
#frame, frame_low, frame_vert, percentage, percentage_low = t1.get_return_value()
plt.plot(frame, percentage, 'r.')
plt.plot(frame_low ,percentage_low, 'b.')
for i in range(len(frame_vert)):
    plt.axvline(frame_vert[i], color = 'black')
    i += 1
plt.ylabel('%')
plt.xlabel('frame/time')
plt.title("Movement Percentage")
if not frame_low:
    plt.ylim(5, 100)
plt.show()