import numpy as np
import cv2
import math


video = cv2.VideoCapture(0)
accumWeight = 0.5
bg = None

video.set(3, 700)
video.set(4, 480)
top, right, bottom, left = 10, 350, 225, 590

num_frames = 0

background_subtractor = cv2.createBackgroundSubtractorMOG2(0, 50, detectShadows=False)

def define_roi(frame):
    roi = frame[top:bottom, right:left]
    return roi

def background_sub(frame):
    fgmask = background_subtractor.apply(frame,learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.bitwise_and(frame, frame, mask=fgmask)
    return fgmask

def image_preprocessing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (7, 7), 0)
    # ret, thresh = cv2.threshold(gaussian, 60, 255, cv2.THRESH_BINARY)
    return gaussian

def hand_contours(frame):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), frame)
    thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    image, contours, hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        cnt = max(contours, key=cv2.contourArea)
        return (thresholded, cnt)

def polygon(cnt, clone):
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(clone, [approx], -1, (0, 255, 0), 1)



def convexHull(cnt):
    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(clone,start,end,[255,0,0],2)
        #cv2.circle(clone,far,5,[0,0,255],-1)

        count_defects = 0

        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(clone, far, 5 ,[0,0,255], -1)

    if count_defects == 1:
        cv2.putText(clone,"2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 2:
        cv2.putText(clone, "3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    elif count_defects == 3:
        cv2.putText(clone,"4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(clone,"5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(clone,"0", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
    extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
    extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
    extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])

    # def draw_extreme_points(clone):
    #
    #     cv2.circle(clone, extLeft, 8, (0, 0, 255), -1)
    #     cv2.circle(clone, extRight, 8, (0, 255, 0), -1)
    #     cv2.circle(clone, extTop, 8, (255, 0, 0), -1)
    #     cv2.circle(clone, extBot, 8, (255, 255, 0), -1)

def drawing(clone):
    cv2.drawContours(clone, [cnt], 0, (0, 255, 0), 2)
    cv2.rectangle(clone, (left, top), (right,bottom), (255, 255, 0), 2)

def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, accumWeight)


while(True):
    ret,frame = video.read()
    frame = cv2.flip(frame, 1)
    # (height, width) = frame.shape[:2]
    clone = frame.copy()

    roi = define_roi(frame)
    #fg_frame = background_sub(clone)
    thresh = image_preprocessing(roi)

    if num_frames < 30:
        run_avg(thresh, accumWeight)
        if num_frames == 1:
            print "[STATUS] please wait! calibrating..."
        elif num_frames == 29:
            print "[STATUS] calibration successfull..."
    else:
        hand = hand_contours(thresh)
        if hand is not None:
            (thresholded, cnt) = hand
            convexHull(cnt)
            cv2.drawContours(clone, [cnt + (right, top)], -1, (0, 255, 0), 2)
            cv2.imshow("Thesholded", thresholded)

    cv2.rectangle(clone, (left, top), (right,bottom), (255, 255, 0), 2)
        #drawing(clone)
    num_frames += 1
    # convexHull(cnt)
    # drawing(clone)

    #polygon(cnt, clone)

    #cv2.imshow("background", fg_frame)
    cv2.imshow("thresh", clone)

    if 0xFF & cv2.waitKey(5) == 27:
            break

video.release()
cv2.destroyAllWindows()
