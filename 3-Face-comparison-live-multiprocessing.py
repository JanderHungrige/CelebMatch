#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:56:49 2020

@author: base
"""

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
import math 
#from imutils import face_utils
#import dlib
import pandas as pd
from pathlib import Path
from numpy import expand_dims
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from time import time

from multiprocessing import Process



# DEFINE VARIABLES
#landmark_predictor = dlib.shape_predictor('/home/base/Documents/Git/Projekte/Face-celeb-rec/shape_predictor_68_face_landmarks.dat')

face_cascade = cv2.CascadeClassifier(str(Path.cwd() / 'haarcascade_frontalface_alt.xml'))
brt = 90  # value could be + or - for brightness or darkness
cap = cv2.VideoCapture(0)
gray=False
p=35# frame size around detected face
width=height=224 # size of the cropped image. Same as required for network
mitte=np.empty(shape=[0, 0])
mittleres_Gesicht_X=()

resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                                pooling='avg')  # pooling: None, avg or max


EMBEDDINGS_Celebs=pd.read_json(Path.cwd() / 'EMBEDDINGS_8k.json')
# Devide Embeddings into 4 parts to be used with 4 parallel threads
Divisor=np.floor_divide(len(EMBEDDINGS_Celebs.File),4)

EMBEDDINGS_Celebs_a=EMBEDDINGS_Celebs[:Divisor-1]
EMBEDDINGS_Celebs_b=EMBEDDINGS_Celebs[Divisor:2*Divisor-1]
EMBEDDINGS_Celebs_c=EMBEDDINGS_Celebs[2*Divisor:3*Divisor-1]
EMBEDDINGS_Celebs_d=EMBEDDINGS_Celebs[3*Divisor:]


def compare_embeddings(EMBEDDINGS, EMBEDDINGS_Celebs):
    EuDist=[]#np.empty(len(EMBEDDINGS_Celebs.File),1)
    for i in range(len(EMBEDDINGS_Celebs.File)):
        Celebs=np.array(EMBEDDINGS_Celebs.Embedding[i]) 
        EuDist.append(np.linalg.norm(EMBEDDINGS-Celebs))
        minDist=np.argmin(EuDist)
        folder_idx= EMBEDDINGS_Celebs.Name[np.argmin(EuDist)]
        image_idx = EMBEDDINGS_Celebs.File[np.argmin(EuDist)] 

    return minDist


def find_index(EMBEDDINGS_Celebs,EuDist_a,EuDist_b,EuDist_c,EuDist_d):
                    image_idx = EMBEDDINGS_Celebs.File[np.argmin(EuDist)] 


TODO:
    MINdIST, FOLDER_INDEX ... IN PANDAS UND RETURN
    EINLESEN IN FIND_INDEX
    VERGLEICHE WO MIN
    FINDE GLOBALEN INDEX, BZW GIB NAMEN DES SETS NOCH MIT IN PANDAS, FINDE WELCHES SET GEWINNT. EXTRAHIERE NAMEN UND BILD ETC

ret, frame = cap.read() 
framemitte=np.shape(frame)[1]/2

while(True):
# CAPTURE FRAME BY FRAME
    
    ret, frame = cap.read() 
    if gray==True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    cv2.imshow('frame', frame) 

    

#DECTECT FACE IN VIDEO CONTINUOUSLY       
    faces_detected = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces_detected:
        rechteck=cv2.rectangle(frame, (x-p, y-p+2), (x+w+p, y+h+p+2), (0, 255, 0), 2)  
        rechteck=cv2.rectangle(frame, (x-p, y-p+2), (x+int(np.ceil(height))+p, y+int(np.ceil(height))+p+2), (0, 0, 255), 2)  

        cv2.imshow('frame', rechteck)     

# DETECT KEY INPUT  - ESC OR FIND MOST CENTERED FACE  
    key = cv2.waitKey(1)
    if key == 27: #Esc key
        cap.release()
        cv2.destroyAllWindows()
        break
    if key & 0xFF == ord('s'): 
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
            img=frame[y-p+2:y+h+p-2, x-p+2:x+w+p-2] #use only the detected face; crop it +2 to remove frame
            #cv2.destroyAllWindows()            
            cv2.imshow('frame', img)
            end2=time()
            print('detect face ',end2-start2)
# DETECT LANDMARKS 
#             outlines=landmark_predictor(frame,faces_detected[mittleres_Gesicht_X])
#             outlines = face_utils.shape_to_np(outlines)
# CROP IMAGE 
            if img.shape > (width,height):
                start3=time()
                print('size good')
                img_small=cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST) #resize the image to desired dimensions e.g., 256x256  
                end3=time()
                print('face crop', end3-start3)
#CREATE FACE EMBEDDINGS
                start4=time()
                # Make images the same as they were trained on in the VGGface2 Model
                img_small = img_small.astype('float32')
                img_small = expand_dims(img_small, axis=0)
                # prepare the face for the model, e.g. center pixels
                img_small = preprocess_input(img_small, version=2)
                EMBEDDINGS = resnet50_features.predict(img_small)
                #print('.')
                end4=time()
                print('create face embeddings' , end4-start4)
# READ CELEB EMBEDDINGS AND COMPARE  
                start5=time()
                #EuDist=[]
                EuDist,folder_idx=compare_embeddings(EMBEDDINGS,EMBEDDINGS_Celebs)

                image_idx = EMBEDDINGS_Celebs.File[np.argmin(EuDist)] 
                print(' Distance value: ', np.argmin(EuDist), ' | ' , 'Name: ', EMBEDDINGS_Celebs.Name[np.argmin(EuDist)],' | ' ,' Filename: ', EMBEDDINGS_Celebs.File[np.argmin(EuDist)])
                end5=time()
                print('find facematch', end5-start5)
# PLOT IMAGES       
                start6=time()
                path=Path.cwd()
                pfad=str(Path.cwd() / 'Celebs' / str(folder_idx) / str(image_idx))
                Beleb=cv2.imread(pfad)
                
                Beleb=cv2.resize(Beleb, (np.shape(img)[0] ,np.shape(img)[1]), interpolation=cv2.INTER_NEAREST)
                if np.shape(img) == np.shape(Beleb):
                    numpy_horizontal = np.hstack((img, Beleb))
                else:
                    print('celeb image not large enough')
                
                cv2.namedWindow('thats_you',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('thats_you',5*width,5*height)
                cv2.imshow('thats_you', numpy_horizontal)   
                end6=time()
                print('print create image', end6-start6)
# CLEARING ALL VARIANLES AND CONTINUE WITH THE PROGRAM
                total=time()
                print('totaltime ', total-start1)


                cv2.waitKey(0) #hit any key
                faces_detected=None
                mittleres_Gesicht_X=None        
                end6=time()
                print('print create image', end6-start6)
                img=None
                img_small=None
                pixels=None
                samples=None
                EMBEDDINGS=None
           
                cv2.destroyWindow('thats_you')
                #cv2.destroyAllWindows()
            else: 
                print('image to small')
                faces_detected=None
                img=None
                img_small=None
                pixels=None
                samples=None
                EMBEDDINGS=None
                mittleres_Gesicht_X=None
                mitte=None

                 

            
            if key == 27: #Esc key
                break
#             if key == 13: #Enter key
#                 cap.release()
#                 cv2.destroyAllWindows()

                
        else:
            print('noface detected')
            
        

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

# Now detect face and crop

