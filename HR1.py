'''
Created on Jan 19, 2016

@author: sikarwar
'''
import cv2
import datetime
import numpy as np

cap = cv2.VideoCapture(0)
lastFrame = None
while True:
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if lastFrame is None:
        lastFrame = gray
        continue
    
    blur = cv2.GaussianBlur(gray,(21,21),0)
    frameDelta = cv2.absdiff(lastFrame, gray)
    thresh = cv2.threshold(frameDelta, 15, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.dilate(thresh, None, iterations = 2)
    
    #ret,thresh = cv2.threshold(frameDelta,20,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        if cv2.contourArea(c) < 500:
            continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text= 'Occupied'
    
        cv2.putText(frame, 'Room Status :{} '.format(text), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
        cv2.putText(frame, datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'),
                (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255),1)
    

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('thresh', thresh)
    cv2.imshow('frameDelta', frameDelta)
    
    lastFrame = gray
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()