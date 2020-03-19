#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:47:36 2020

@author: base
"""

import numpy as np
import cv2

#face_cascade = cv2.CascadeClassifier('./data/lbpcascades/lbpcascade_frontalface.xml')
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')                        
#a=np.empty(20)
a=[]
#d=np.empty(20)
cap  = cv2.VideoCapture(0)                            
while(True):
    ret, frame =cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.array(frame, dtype='uint8')         
    faces_detected = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=10)
    if faces_detected is not None:                                                      
        a.append(np.shape(faces_detected)[0])                                                                          
        b=sum(a)/len(a) 
        print(b)                                               
        if len(a)==20:                                                                 
            a=[sum(a)/len(a)] 
            
                                    
 
    #key = cv2.waitKey(1)

    


cv2.destroyAllWindows()
cap.release()
