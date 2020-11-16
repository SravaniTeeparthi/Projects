#generates groundtruth for testing image fro calculating accuracies
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




img = cv2.imread('../data/images/Writing2Round2Side_frame_0.png',cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape[0:2]


#Take all anchor points and validating them
anchor_points= anchor_points(img,10) #define space for rows and colums
BoxCols_lst,BoxRows_lst=get_anchor_boxes(img,170) #define unitspace
valid_anchor_points=valid_anchor_points(img,anchor_points,BoxRows_lst[3],BoxCols_lst[3]) #taking only valid boxes

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
    print(iou)
    return iou

#return whether it is face or noface based on threshold
def BoxGroundTruth(BoxCoords, FaceCoords_1,FaceCoords_2,FaceCoords_3, Th):
    iou1 = iou_ratio(BoxCoords,FaceCoords_1)
    iou2 = iou_ratio(BoxCoords,FaceCoords_2)
    iou3 = iou_ratio(BoxCoords,FaceCoords_3)

    if iou1>=Th or iou2>=Th or iou3>=Th: # checks all the threshold values
        return "face1", BoxCoords
    else:
        return "nonface",BoxCoords

def ClassifyImage(img,FaceCoords_1,FaceCoords_2,FaceCoords_3,AnchorPoints,BoxRows,BoxCols,Th,NumOfBoxes,file,r):
        Num_FaceBoxes=0
        Num_NonFaceBoxes=0
        face_boxes=[]
        non_face_boxes=[]
        for i in range(0,len(AnchorPoints)-1):
            point=AnchorPoints[i-1]
            (a,b,w,h)=(point[0]-(BoxCols[3]/2),point[1]-(BoxRows[3]/2),point[0]+(BoxCols[3]/2),point[1]+(BoxRows[3]/2))
            f_nf,box_coords= BoxGroundTruth((a,b,w,h),FaceCoords_1,FaceCoords_2,FaceCoords_3,Th)

            if f_nf=="face1" and Num_FaceBoxes < (NumOfBoxes/2):
                crop_img_face=img[int(box_coords[1]):int(box_coords[3]),int(box_coords[0]):int(box_coords[2])]
                #img_path="images/face_img/"+file+'_'+str(r)+"_"+str(cur_samples)+".png"
                #cv2.imwrite(img_path,crop_img_face)
                if(crop_img_face.shape==(BoxRows[3],BoxCols[3])): # number
                    face_boxes.append(crop_img_face)
                    Num_FaceBoxes=Num_FaceBoxes+1

            elif f_nf=="nonface" and Num_NonFaceBoxes < (NumOfBoxes/2):
                crop_img_noface=img[int(box_coords[1]):int(box_coords[3]),int(box_coords[0]):int(box_coords[2])]
                #img_path="images/noface_img/"+file+'_'+str(r)+"_"+str(cur_samples)+".png"
                #cv2.imwrite(img_path,crop_img_noface)
                if(crop_img_noface.shape==(BoxRows[3],BoxCols[3])): #number
                    non_face_boxes.append(crop_img_noface)
                    Num_NonFaceBoxes=Num_NonFaceBoxes+1

        # Saving face boxes
        if len(face_boxes) > 1:
            for cur_fb_idx, cur_fb in enumerate(face_boxes):
                face_path="test_files/face_2_2/"+file+"_"+str(r)+ "_" + str(cur_fb_idx) +".npy"
                np.save(face_path,cur_fb)
        # Saving non face boxes
        if len(non_face_boxes) > 1:
            for cur_non_fb_idx, cur_non_fb in enumerate(non_face_boxes):
                nonface_path="test_files/noface_2_2/"+file+"_"+str(r)+ "_" + str(cur_non_fb_idx) +".npy"
                np.save(nonface_path,cur_non_fb)

        return (a,b,w,h),Num_FaceBoxes,Num_NonFaceBoxes


#reads csv files and returns image and csv file
#change based on file names
def read_csvfile(file_name,i):
    path="../data/csv/"
    img_path="../data/images/"
    path=path+file_name
    df=pd.read_csv(path)
    img_path=img_path+file_name.split('_')[0]+"_frame_"+str(i)+".png"
    img=cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return df,img


#declare test files
files=["Writing2Round2Side_Face1.csv"]
for file in files:
    print(file)
    for i in range(0,700,1):
        df,img= read_csvfile(file,i)
        df1,img=read_csvfile(file.split('_')[0]+'_Face2'+".csv",i)
        df2,img=read_csvfile(file.split('_')[0]+'_Face3'+".csv",i)
        print(i)
        if img is not None:
            df  =(df.loc[i][0],df.loc[i][1],df.loc[i][0]+df.loc[i][2],df.loc[i][1]+df.loc[i][3])
            df1 =(df1.loc[i][0],df1.loc[i][1],df1.loc[i][0]+df1.loc[i][2],df1.loc[i][1]+df1.loc[i][3])
            df2 =(df2.loc[i][0],df2.loc[i][1],df2.loc[i][0]+df2.loc[i][2],df2.loc[i][1]+df2.loc[i][3])
            BoxCoords,Num_FaceBoxes,Num_NonFaceBoxes=ClassifyImage(img,df,df1,
                                                                    df2,valid_anchor_points,BoxRows_lst,
                                                                    BoxCols_lst,
                                                                    0.5,1500,file.split('_')[0],i)
