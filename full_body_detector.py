import numpy as np
import cv2
import time
import argparse
import sys
import os

full_body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
upper_body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
lower_body_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')

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
else:
    cap = cv2.VideoCapture(0)


background_subtractor = cv2.createBackgroundSubtractorMOG2()
cap.set(3, 640)
cap.set(4, 480)

while(True):
    ret, frame = cap.read()
    background_subtractor.apply(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    full_body = full_body_cascade.detectMultiScale(gray, 1.1, 4)

    for(i,(x,y,w,h)) in enumerate(full_body):
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
        cv2.putText(frame, 'Person' + " #{}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        upper_body = upper_body_cascade.detectMultiScale(roi_gray)
        for (ubx,uby,ubw,ubh) in upper_body:
            cv2.rectangle(roi_color,(ubx,uby),(ubx+ubw,uby+ubh),(0,255,0),2)

        lower_body = lower_body_cascade.detectMultiScale(roi_gray)
        for (lbx,lby,lbw,lbh) in lower_body:
            cv2.rectangle(roi_color, (lbx,lby), (lbx+lbw, lby+lbh),(0,0,255),2)

    cv2.imshow('Full Body Detector',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
