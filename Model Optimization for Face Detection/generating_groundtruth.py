#import libraries
import numpy as np
import pdb
import cv2
import random
import csv
import pickle
import math
import pandas as pd
from generate_anchorboxes import valid_anchor_points,anchor_points,get_anchor_boxes

img3 = cv2.imread('../data/images/Eating1Round1Side.png',cv2.IMREAD_UNCHANGED)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
h, w = img3.shape[0:2]
#print(h,w)

#Take all anchor points and validating them
anchor_points= anchor_points(img3,7) #space between rows and columns
BoxCols_lst,BoxRows_lst=get_anchor_boxes(img3,170) # define unit space
valid_anchor_points=valid_anchor_points(img3,anchor_points,BoxRows_lst[3],BoxCols_lst[3]) # validating anchor points

#gives iou ratio
def iou_ratio(BoxCoords,FaceCoords):
    xi1 = max(BoxCoords[0],FaceCoords[0])
    yi1 = max(BoxCoords[1],FaceCoords[1])
    xi2 = min(BoxCoords[2],FaceCoords[2])
    yi2 = min(BoxCoords[3],FaceCoords[3])
    inter_area = max(yi2-yi1,0)*max(xi2-xi1,0)
    box1_area = (BoxCoords[3]-BoxCoords[1])*(BoxCoords[2]-BoxCoords[0])
    box2_area = (FaceCoords[3]-FaceCoords[1])*(FaceCoords[2]-FaceCoords[0])
    union_area = box1_area+box2_area-inter_area
    iou = inter_area/union_area
    #print(box1_area,box2_area)
    return iou


#return whether it is face or noface based on threshold
def BoxGroundTruth(BoxCoords, FaceCoords_1,FaceCoords_2,FaceCoords_3, Th):
    iou1 = iou_ratio(BoxCoords,FaceCoords_1)
    iou2 = iou_ratio(BoxCoords,FaceCoords_2)
    iou3 = iou_ratio(BoxCoords,FaceCoords_3)
    if iou1>=Th or iou2>=Th or iou3>=Th:
        return "face1", BoxCoords
    else:
        return "nonface",BoxCoords

#BoxGroundTruth
def BuildTrainData(img,FaceCoords_1,FaceCoords_2,FaceCoords_3,AnchorPoints,BoxRows,BoxCols,Th,NumOfBoxes,file,r):
    Num_FaceBoxes=0
    Num_NonFaceBoxes=0
    Max_samples=3 #define number of boxes needed
    cur_samples=0
    face_boxes=[]
    non_face_boxes=[]
    while(cur_samples < Max_samples):
        i=random.randint(0,len(AnchorPoints)) # choosing random point
        point=AnchorPoints[i-1]
        #defining boxes for all aspect ratios and scales

        (a1,b1,w1,h1)=(point[0]-(BoxCols[1]/2),point[1]-(BoxRows[1]/2),point[0]+(BoxCols[1]/2),point[1]+(BoxRows[1]/2))
        (a2,b2,w2,h2)=(point[0]-(BoxCols[2]/2),point[1]-(BoxRows[2]/2),point[0]+(BoxCols[2]/2),point[1]+(BoxRows[2]/2))
        (a3,b3,w3,h3)=(point[0]-(BoxCols[3]/2),point[1]-(BoxRows[3]/2),point[0]+(BoxCols[3]/2),point[1]+(BoxRows[3]/2))
        #checking if face or noface
        f_nf,box_coords= BoxGroundTruth((a3,b3,w3,h3),FaceCoords_1,FaceCoords_2,FaceCoords_3,Th)
        #if face
        if f_nf=="face1" and Num_FaceBoxes < (NumOfBoxes/2):
            #cv2.rectangle(img,(int(a),int(b)),(int(w),int(h)),(255,255,255),2)
            #cv2.rectangle(img,(int(FaceCoords_1[0]),int(FaceCoords_1[1])),(int(FaceCoords_1[2]),int(FaceCoords_1[3])),(0,0,255),2)
            #cv2.rectangle(img,(int(FaceCoords_2[0]),int(FaceCoords_2[1])),(int(FaceCoords_2[2]),int(FaceCoords_2[3])),(0,0,255),2)
            #cv2.rectangle(img,(int(FaceCoords_3[0]),int(FaceCoords_3[1])),(int(FaceCoords_3[2]),int(FaceCoords_3[3])),(0,0,255),2)
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            #import pdb; pdb.set_trace()
            crop_img_face=img[int(box_coords[1]):int(box_coords[3]),int(box_coords[0]):int(box_coords[2])]
            img_path="images/face_2_2_img/"+file+'_'+str(r)+"_"+str(cur_samples)+".png"
            cv2.imwrite(img_path,crop_img_face)
            if(crop_img_face.shape==(BoxRows[3],BoxCols[3])): # number
                face_boxes.append(crop_img_face)
                Num_FaceBoxes=Num_FaceBoxes+1
        #if not a face
        elif f_nf=="nonface" and Num_NonFaceBoxes < (NumOfBoxes/2):
            crop_img_noface=img[int(box_coords[1]):int(box_coords[3]),int(box_coords[0]):int(box_coords[2])]
            img_path="images/noface_2_2_img/"+file+'_'+str(r)+"_"+str(cur_samples)+".png"
            cv2.imwrite(img_path,crop_img_noface)
            if(crop_img_noface.shape==(BoxRows[3],BoxCols[3])): #number
                non_face_boxes.append(crop_img_noface)
                Num_NonFaceBoxes=Num_NonFaceBoxes+1

        cur_samples = min(Num_NonFaceBoxes, Num_FaceBoxes) # loop ends when number of cur=samples=Max_samples

    non_face_boxes=non_face_boxes[:Max_samples]

    # Saving face boxes
    if len(face_boxes) > 1:
        for cur_fb_idx, cur_fb in enumerate(face_boxes):
            face_path="face_2_2/"+file+"_"+str(r)+ "_" + str(cur_fb_idx) +".npy"
            np.save(face_path,cur_fb)
    # Saving non face boxes
    if len(non_face_boxes) > 1:
        for cur_non_fb_idx, cur_non_fb in enumerate(non_face_boxes):
            nonface_path="noface_2_2/"+file+"_"+str(r)+ "_" + str(cur_non_fb_idx) +".npy"
            np.save(nonface_path,cur_non_fb)

    return (a,b,w,h),Num_FaceBoxes,Num_NonFaceBoxes


#used to read csv and iamges for files
def read_csvfile(file_name,i):
    path="../data/csv/"
    img_path="../data/frames/"
    path=path+file_name
    df=pd.read_csv(path)
    pdb.set_trace()
    img_path=img_path+file_name.split('_')[0]+"_frame_"+str(i)+".jpg"
    img=cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return df,img


#training files
files=["Eating1Round1Side_Face1.csv","Eating1Round2Side_Face1.csv",
       "Pointing1Round2Side_Face1.csv","Pointing2Round2Side_Face1.csv",
      "Typing1Round1_Face1.csv","Typing2Round1_Face1.csv",
       "UsingMouse1Round1_Face1.csv","UsingMouse1Round1Side_Face1.csv",
    "Writing1Round2Side_Face1.csv"]
#files=["Typing2Round1_Face1.csv", "UsingMouse1Round1_Face1.csv","UsingMouse1Round1Side_Face1.csv",
#    "Pointing2Round2Side_Face1.csv"]
for file in files:
    print(file)
    for i in range(0,700,29):
        df,img= read_csvfile(file,i)
        df1,img=read_csvfile(file.split('_')[0]+'_Face2'+".csv",i)
        df2,img=read_csvfile(file.split('_')[0]+'_Face3'+".csv",i)
        print(i)
        if img is not None:
            df  =(df.loc[i][0],df.loc[i][1],df.loc[i][0]+df.loc[i][2],df.loc[i][1]+df.loc[i][3])
            df1 =(df1.loc[i][0],df1.loc[i][1],df1.loc[i][0]+df1.loc[i][2],df1.loc[i][1]+df1.loc[i][3])
            df2 =(df2.loc[i][0],df2.loc[i][1],df2.loc[i][0]+df2.loc[i][2],df2.loc[i][1]+df2.loc[i][3])
            BoxCoords,Num_FaceBoxes,Num_NonFaceBoxes=BuildTrainData(img,df,df1,
                                                                    df2,
                                                                    valid_anchor_points,BoxRows_lst,BoxCols_lst,
                                                                    0.6,1000,file.split('_')[0],i)
            #print(Num_FaceBoxes,Num_NonFaceBoxes)
