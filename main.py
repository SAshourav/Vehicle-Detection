import time

import cv2
import numpy as np

# web camera

cap = cv2.VideoCapture('video.mp4')
count_line_position = 550
mini_width_rec = 80
min_height_width = 80
# Initialize Substructor

algo = cv2.createBackgroundSubtractorMOG2()
frame_delay = 0.01
def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1

    return cx,cy

detect = []

offset = 6 #allowable error between pixel
counter = 0

while True:
    ret,frame1 = cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    # applying on each frame

    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    karnel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, karnel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, karnel)
    counterShape,h = cv2.findContours(dilatada,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(25,count_line_position),(1250,count_line_position),(255,127,0),3)

    # make rectrangle around vehicle

    for(i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=mini_width_rec) and (h>=min_height_width)

        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1, f"Vehicle {counter}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 244, 0))

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter += 1
                cv2.line(frame1,(25,count_line_position),(1250,count_line_position),(0,127,255),3)
                detect.remove((x,y))
                print(f"Vehicle Counter {counter}")

    cv2.putText(frame1,f"Vehicle Counter {counter}",(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255))
    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:
        break

    time.sleep(frame_delay)

cv2.destroyWindow('Video Original') # closing the window
cap.release()
