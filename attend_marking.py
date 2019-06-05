# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:41:20 2019

@author: senthilku
"""

# import required packages
import cv2
import dlib
#import time

# load input image
image = cv2.imread('img1.jpg')
cv2.imshow('img',image)
image1=image

cnnFaceDetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

faces_cnn = cnnFaceDetector(image, 1)
f1=[]

for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y
    f1.append(image[y:y+h,x:x+w,:])
    cv2.rectangle(image1, (x,y), (x+w,y+h), (0,0,255), 2)

cv2.imshow("All Face Images",image1)
#cv2.imshow("face", f1[3])
#cv2.imwrite('per1.jpg',f1[0])
#cv2.imwrite('per2.jpg',f1[1])
#cv2.imwrite('per3.jpg',f1[2])
#cv2.imwrite('per4.jpg',f1[3])
#cv2.imwrite('per5.jpg',f1[4])
#cv2.imwrite('per6.jpg',f1[5])
#cv2.imwrite('per7.jpg',f1[6])
#cv2.imwrite('per8.jpg',f1[7])
#cv2.imwrite('per9.jpg',f1[8])
#cv2.imwrite('per10.jpg',f1[9])
#fa=f1[0]

import face_recognition
loc=face_recognition.face_locations(image,number_of_times_to_upsample=1,model="cnn")
known_image = face_recognition.load_image_file("img1.jpg",)
face_encoding = face_recognition.face_encodings(known_image,known_face_locations=loc)
import os
import glob
import re
pre=0;
abse=0;
c1=[]
roll_num=[]
for filename in glob.glob("D:/Face_OpenCV/students_image/*.jpg"):
    #im1=cv2.imread(filename)
    check_image = face_recognition.load_image_file(filename)
    check_image=cv2.resize(check_image,(0,0), fx=0.2, fy=0.2)
    #cv2.imshow('Query Image',check_image)
    #cv2.waitKey(0)
    loc1=face_recognition.face_locations(check_image,number_of_times_to_upsample=1,model="cnn")
    face_encoding_check = face_recognition.face_encodings(check_image,known_face_locations=loc1)

    results1=[]
    for i in range(len(face_encoding)):
        cnt=0
        results = face_recognition.compare_faces([face_encoding[i]], face_encoding_check[0],tolerance=0.5)
        results1.append(results[0])
        #print(results)
        if results[0] == True:
            cnt=cnt+1
            top, right, bottom, left = loc[i]
            match_image = image[top:bottom, left:right]
            #cv2.imshow('matched_image',match_image)
            #cv2.waitKey(0)
            
#            if cnt==1:
#                print(filename,"is present")
#                if cnt==0:
#            print(filename, "is absent")
                
    c1.append(results1.count(True))
    if results1.count(True)>0:
        print(filename,"is Present")
        pre=pre+1
    else:
        print(filename, "is absent") 
        abse=abse+1
    #roll_num.append(re.findall('\d+',filename))

print('Total Number of Students Present = ',pre)
print('Total Number of Students Absent = ',abse)

#num=[]
#for i in range(len(roll_num)):
#    num.append(roll_num[i][0])
    
import pandas as pd
import numpy as np
import datetime

sheet=pd.read_csv('attendance_batch5.csv')
date=datetime.datetime.now()
dat=date.strftime("%d-%m-%Y") 
sheet[dat]=c1

#file=pd.DataFrame({'Roll_No':num,dat:c1},columns=['Roll_No',dat],index=np.arange(1,13))
sheet.to_csv('attendance_batch5.csv',index=None)


