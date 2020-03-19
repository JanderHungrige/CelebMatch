#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 21:44:09 2020

@author: base
"""

import numpy as np
import cv2
import os
#import time
#from imutils import face_utils
#import dlib
import pandas as pd
from pathlib import Path
from numpy import expand_dims
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from time import time

# DEFINE VARIABLES
#landmark_predictor = dlib.shape_predictor('/home/base/Documents/Git/Projekte/Face-celeb-rec/shape_predictor_68_face_landmarks.dat')



# Set Opencv-gstreamer pipeline

buildinfo = cv2.getBuildInformation()

if buildinfo.find("GStreamer") < 0:
    print('no GStreamer support in OpenCV')
    exit(0)

#can be changed to e.g. 640x480
#width=1024
#height=760

width=1280
height=800

#Set g stremaer pipeline
cmd = "v4l2-ctl -d /dev/video0 --set-fmt-video=pixelformat=GRBG,width={width},height={height}".format(width=width, height=height)
os.system(cmd)
cmd = "v4l2-ctl -d /dev/video0 --set-selection=target=crop,left=0,top=4,width={width},height={height}".format(width=width, height=height)
os.system(cmd)
#cmd = "v4l2-ctl -d /dev/video0 --set-selection=target=crop,left={width}-320,top={height}-164+4,width={width},height={height}".format(width=width, height=height)
#os.system(cmd)

#cmd = "v4l2-ctl -d /dev/video0 --set-ctrl=horizontal_skipping=1 " 
#os.system(cmd)
#cmd = "v4l2-ctl -d /dev/video0 --set-ctrl=vertical_skipping=1 " 
#os.system(cmd)
#cmd = "v4l2-ctl -d /dev/video0 --set-ctrl=row_binning=1" 
#os.system(cmd)
#cmd = "v4l2-ctl -d /dev/video0 --set-ctrl=col_binning=1 " 
#os.system(cmd)

pipeline = "v4l2src device=/dev/video0 ! video/x-bayer,format=grbg,depth=8,width={width},height={height} ! bayer2rgb ! videoconvert !videoscale ! video/x-raw,width=640,height=400 !  appsink".format(width=width, height=height)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('VideoCapture not opened')
    exit(0)

#face_cascade = cv2.CascadeClassifier(str(Path.cwd() / 'haarcascade_frontalface_alt.xml'))
#face_cascade = cv2.CascadeClassifier(str(Path.cwd() / 'lbpcascade_frontalface.xml'))
face_cascade = cv2.CascadeClassifier(str(Path.cwd() / 'lbpcascade_frontalface_improved.xml'))


brt = 90  # value could be + or - for brightness or darkness
gray=False
p=35# frame size around detected face
width=height=224 # size of the cropped image. Same as required for network
mitte=np.empty(shape=[0, 0])
mittleres_Gesicht_X=()

resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                                pooling='avg')  # pooling: None, avg or max


EMBEDDINGS_Celebs=pd.read_json(Path.cwd() / 'EMBEDDINGS_8k.json')



ret, frame = cap.read() 
framemitte=np.shape(frame)[1]/2


##Get the openCV window up front...does not work
#cv2.namedWindow("GetFocus", cv2.WINDOW_NORMAL);
#bild = np.zeros((256, 256, 1), dtype = "uint8")
##Mat, bild = cv2.Mat.zeros(100, 100, CV_8UC3);
#cv2.imshow("GetFocus", bild);
#cv2.setWindowProperty("GetFocus", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
#cv2.waitKey(1) 
#cv2.setWindowProperty("GetFocus", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL);
#cv2.moveWindow('GetFocus',200,200)
#cv2.destroyWindow("GetFocus");


while(True):
# CAPTURE FRAME BY FRAME
    
    ret, frame = cap.read() 
    if gray==True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame=cv2.flip(frame,1)   
    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', frame) 

    
#DECTECT FACE IN VIDEO CONTINUOUSLY       
    faces_detected = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)#, Size(50,50))
    for (x,y,w,h) in faces_detected:
        rechteck=cv2.rectangle(frame, (x-p, y-p+2), (x+w+p, y+h+p+2), (0, 255, 0), 2)  
        #rechteck=cv2.rectangle(frame, (x-p, y-p+2), (x+int(np.ceil(height))+p, y+int(np.ceil(height))+p+2), (0, 0, 100), 2)  
        cv2.imshow('frame', rechteck)     

# DETECT KEY INPUT  - ESC OR FIND MOST CENTERED FACE  
    key = cv2.waitKey(1)
    if key == 27: #Esc key
        cap.release()
        cv2.destroyAllWindows()
        break
   # if key & 0xFF == ord('s'): 
    if key ==32: 
        mittleres_Gesicht_X=()
        mitte=()
        #faces_detected = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        if faces_detected != (): # only if the cascader detected a face, otherwise error
            start1 = time()
            for (x,y,w,h) in faces_detected:
                mitte=np.append(mitte,(x+w/2))
                
            mittleres_Gesicht_X = (np.abs(mitte - framemitte)).argmin()
            end1 = time()
            print('detect middel face ', end1-start1)
# DETECT FACE
            start2=time()
            print(faces_detected[mittleres_Gesicht_X])
            (x, y, w, h) = faces_detected[mittleres_Gesicht_X]
            img=frame[y-p+2:y+h+p-2, x-p+2:x+w+p-2] #use only the detected face; crop it +2 to remove frame # CHECK IF IMAGE EMPTY (OUT OF IMAGE = EMPTY)     

            if len(img) != 0: # Check if face is out of the frame, then img=[], throwing error
                end2=time()
                print('detect face ',end2-start2)
# DETECT LANDMARKS 
#             outlines=landmark_predictor(frame,faces_detected[mittleres_Gesicht_X])
#             outlines = face_utils.shape_to_np(outlines)
# CROP IMAGE 
                start3=time()
                if img.shape > (width,height): #downsampling
                    img_small=cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) #resize the image to desired dimensions e.g., 256x256  
                elif img.shape < (width,height): #upsampling
                    img_small=cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC) #resize the image to desired dimensions e.g., 256x256                      
                cv2.imshow('frame',img_small)
                cv2.waitKey(1) #hit any key
                end3=time()
                print('face crop', end3-start3)
#CREATE FACE EMBEDDINGS
                start4=time()
                # Make images the same as they were trained on in the VGGface2 Model
                pixels = img_small.astype('float32')
                samples = expand_dims(pixels, axis=0)
                # prepare the face for the model, e.g. center pixels
                samples = preprocess_input(samples, version=2)
                EMBEDDINGS = resnet50_features.predict(samples)
                #print('.')
                end4=time()
                print('create face embeddings' , end4-start4)
# READ CELEB EMBEDDINGS AND COMPARE  
                start5=time()
                # print(len(EMBEDDINGS.File))
                EuDist=[]
                for i in range(len(EMBEDDINGS_Celebs.File)):
                    Celebs=np.array(EMBEDDINGS_Celebs.Embedding[i]) 
                    dist = np.linalg.norm(EMBEDDINGS-Celebs)
                    EuDist.append(np.linalg.norm(EMBEDDINGS-Celebs))
                folder_idx= EMBEDDINGS_Celebs.Name[np.argmin(EuDist)]
                image_idx = EMBEDDINGS_Celebs.File[np.argmin(EuDist)] 
                print(' Distance value: ', np.argmin(EuDist), ' | ' , 'Name: ', EMBEDDINGS_Celebs.Name[np.argmin(EuDist)],' | ' ,' Filename: ', EMBEDDINGS_Celebs.File[np.argmin(EuDist)])
                end5=time()
                print('find facematch', end5-start5)
# PLOT IMAGES       
                start6=time()
                path=Path.cwd()
                pfad=str(Path.cwd() / 'Celebs' / str(folder_idx) / str(image_idx))
                Beleb=cv2.imread(pfad)                  
                if np.shape(Beleb) != (width,height): 
                    Beleb=cv2.resize(Beleb, (np.shape(img_small)[0] ,np.shape(img_small)[1]), interpolation=cv2.INTER_AREA)
                numpy_horizontal = np.hstack((img_small, Beleb))
                cv2.namedWindow(EMBEDDINGS_Celebs.Name[np.argmin(EuDist)],cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(EMBEDDINGS_Celebs.Name[np.argmin(EuDist)], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                #cv2.resizeWindow('thats_you',5*width,5*height)
                numpy_horizontal= cv2.putText(numpy_horizontal, EMBEDDINGS_Celebs.Name[np.argmin(EuDist)], (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (116, 161, 142), 1)
                cv2.imshow(EMBEDDINGS_Celebs.Name[np.argmin(EuDist)], numpy_horizontal)   

                end6=time()
                print('print create image', end6-start6)
# CLEARING ALL VARIANLES AND CONTINUE WITH THE PROGRAM
                total=time()
                print('totaltime ', total-start1)


                cv2.waitKey(0) #hit any key
                faces_detected=None
                mittleres_Gesicht_X=None        
                img=None
                img_small=None
                pixels=None
                samples=None
                EMBEDDINGS=None
           
                cv2.destroyWindow(EMBEDDINGS_Celebs.Name[np.argmin(EuDist)])
                #cv2.destroyAllWindows()
#        
    
                     
    
                
                if key == 27: #Esc key
                    break
    #             if key == 13: #Enter key
    #                 cap.release()
    #                 cv2.destroyAllWindows()

            else: 
                rame= cv2.putText(frame, 'FACE MUST BE IN FRAME', (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (129, 173, 181), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(900)
                
        else:
            print('noface detected')
            
        

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

# Now detect face and crop

