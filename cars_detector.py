import numpy as np
import cv2
import time
import argparse
import sys
import os

car_cascade = cv2.CascadeClassifier('haarcascade_cars.xml')

parser = argparse.ArgumentParser(description='Cars Detection using Cascade Classifier')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file: %s, doesn't exist" %(args.image))
        sys.exit(1)
    cap = cv2.VideoCapture(args.image)
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file: %s, doesn't exist" %(args.video))
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)


while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    cv2.imshow('Cars Detector', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
