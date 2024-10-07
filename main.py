'''
files and their roles
1. main.py (main file that controls the entire application. This is where all other logics come together)
2. tracker.py (contains the tracking algorithm. Responsible for identfying the centers of the bounding boxes and assigning IDs to them)
3. model_training.ipynb (reponsible for training the object detection model)
4. 994.pt (Export model with an accuracy of 99.4%)
5. dataset (Contains the data used to train the model)
6. frames.txt (Contains the number of frames and the center of each bubble
7. Custom_data.yaml (Configuration to the path of the dataset and classes for model training)
'''


# LOAD DEPENDANCIES
from tracker import *  #Tracking algorithm code
from ultralytics import YOLO #For training custom data
import cv2 #For drawing boundary boxs and viewing the video file
import cvzone
import math

import numpy as np #for numerical calculations

video = 'Dot_Track_Vid.mp4' #File to run the object detection and tracking one
highest = '994.pt' #deeplearning model with 99.4% accuracy after training

model = YOLO(highest) #Reading the model into memory using Yolo

class_names = ['circle'] #Name of the class of the circles

tracker = bubble_tracker()  #Loading the tracking algorithm to memory

cap = cv2.VideoCapture(video) #Initializing open cv to read the file

# Setting the width and height of the display window
cap.set(3,1280)
cap.set(4,720)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #Height of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #Width of the video
fps = cap.get(cv2.CAP_PROP_FPS) #Frames Per Second
fourcc = cv2.VideoWriter_fourcc('P','I','M','1') #Encoding video in PIM1 Format
out = cv2.VideoWriter('output.avi', fourcc, fps, (width,  height)) #To begin the video writing

count = 0 #Track the number of frames
while True:
    ret, frame = cap.read() #To read the file passed in
    count += 1 #Tracks and counts the number of frames
    if ret is None:
        break

    results = model(frame, stream=True) #Deep learning model is being used on the frame to detect bubbles. This returns the bounding box coordinates and accuracy of each bubble detected
    detections = []
    for result in results:
        boxes = result.boxes #Accessing the bounding box coordinates

        #Drawing the bounding box for each detected bubble
        for box in boxes:
            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0] #Unpacking the coordinates in x1,y1,x2, and y2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1)) #Draws the bounding box using cvzone

            detections.append([x1,y1,x2, y2]) #The coordinates are pushed into the list. Will be used for object tracking
            # Gets the name of the class
            cls = int(box.cls[0])
            # Gets the Confidence of the prediction
            conf = math.ceil(box.conf[0] * 100) / 100

        boxes_id = tracker.get_data(detections) #Class founding in tracker.py to assign an id to each bubble


        # Logic for drawing the  tracking dot and id boxes
        for box_id in boxes_id:
            x1,y1,x2,y2,id = box_id
            cx = int((x1+x2)/2)
            cy = int((y1 + y2) / 2)
            cvzone.putTextRect(frame, f'{conf} ', (max(0, x1), max(20, y1)), scale=1.5)
            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)
    out.write(frame)
    conf = math.ceil(box.conf[0] * 100) / 100
    # Saving frames and coordinates in frames.txt
    text = ""
    text += str(count)+", "
    for i in range(len(detections)):
        # Center of each bubble
        text += str((int((detections[i][0]+detections[i][2])/2), (int(detections[i][1] + detections[i][3])/2)))+ ', '
    file = open('frame.txt','a')
    file.write(f'{text}\n')
    file.close()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
out.release()