import cv2
import numpy as np

# web camera

cap = cv2.VideoCapture('video.mp4')
count_line_position = 550
mini_width_rec = 80
min_height_width = 80
# Initialize Substructor

algo = cv2.createBackgroundSubtractorMOG2()



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




    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyWindow('Video Original') # closing the window
cap.release()
