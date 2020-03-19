#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:35:23 2020

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
import pandas as pd
from pathlib import Path
from numpy import expand_dims
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from time import time
import multiprocessing as mp
import psutil

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('VideoCapture not opened')
    exit(0)

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

def splitDataFrameIntoSmaller(df, chunkSize):
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf
      
def spawn(queue,YourFace, CelebFaces):
    results = []
    procs = list()
    n_cpus = psutil.cpu_count()
    for cpu in range(n_cpus):
        affinity = [cpu]
        d = dict(affinity=affinity)
        p = mp.Process(target=faceembedding, args=[YourFace, CelebFaces, queue], kwargs=d)
        p.start()
        procs.append(p)
    for p in procs:
        results.append(queue.get)
        p.join()
        print('joined')
    return results

def faceembedding(YourFace, CelebFaces, queue, affinity):
    proc = psutil.Process()  # get self pid
    proc.cpu_affinity(affinity)
    print(affinity)
    np.random.seed()
    EuDist=[]
    for i in range(len(CelebFaces)):
        Celebs=np.array(CelebFaces[i]) 
        EuDist.append(np.linalg.norm(YourFace-Celebs))    
#    return EuDist
        queue.put(EuDist)      

#äceleb_embeddings=splitDataFrameIntoSmaller(EMBEDDINGS_Celebs, int(np.ceil(len(EMBEDDINGS_Celebs)/4)))   
    
while(True):
# CAPTURE FRAME BY FRAME
    
    ret, frame = cap.read() 
    if gray==True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame=cv2.flip(frame,1)   
    #####cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    #####cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
    if key ==32: 
        mittleres_Gesicht_X=()
        mitte=()
        if faces_detected != (): # only if the cascader detected a face, otherwise error
            start1 = time()
#FIND MOST MIDDLE FACE            
            for (x,y,w,h) in faces_detected:
                mitte=np.append(mitte,(x+w/2))               
            mittleres_Gesicht_X = (np.abs(mitte - framemitte)).argmin()
            end1 = time()
            print('detect middel face ', end1-start1)
# FRAME THE DETECTED FACE
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
                #jobs=[]
                output = mp.Queue()    
                EuDist = spawn(output, EMBEDDINGS, np.array(EMBEDDINGS_Celebs.Embedding))
                 
#                pool=mp.Pool(processes=4)
#                for x in range(len(celeb_embeddings)):
#                    Werte=celeb_embeddings[x]
#                    results=pool.map(faceembedding, Werte)
                    
#                results = [pool.apply(faceembedding, args=celeb_embeddings[x]) for x in range(len(celeb_embeddings))]
                
                
#                processes = [mp.Process(target=faceembedding, args=(celeb_embeddings[x])) for x in range(len(celeb_embeddings))]
#                jobs=appned(i)
#                
                
                

                    #                for i in range(len(celeb_embeddings)):
#                    Werth=celeb_embeddings[i]
#                    nico = mp.Process(target=faceembedding, args=(Werth,output))
#                    jobs.append(nico)
#                    nico.start()
#                    nico.join()
#                    result=output.get()
#                for i in processes:
#                    i.start()
#                    #jobs.append(processes)
#                for i in processes: 
#                    #processes.start()
#                    i.join()
#                #processes.join()    
             #  EuDist=[output.get() for i in processes]
#                EuDist=output.get()
                    
                #EuDist=faceembedding(EMBEDDINGS_Celebs  )    
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
               # cv2.destroyAllWindows()
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

