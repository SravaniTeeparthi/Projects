#import libraries
import numpy as np
import pdb
import cv2
import random
import csv
import pickle
import math
import pandas as pd



#defining anchor points
def valid_anchor_points(img,anchor_points,bh,bw):
    valid_anchor_points=[]
    for (i,j ) in anchor_points:
        h, w = img.shape[0:2]
        h_r = bh #height
        w_r = bw #width
        if (i-(w_r/2)>0) and (j-(h_r/2)>0) and  (i+(w_r/2)<w) and (j+(h_r/2)<h) :
            valid_anchor_points.append((i,j))
            #cv2.circle(img,(i,j),1,(0,0,0),1)
    #cv2.imshow('image',img)
    #cv2.imwrite('img_validpoints.png',img)
    #cv2.waitKey(0)
    #pdb.set_trace()
    return valid_anchor_points

def anchor_points(img,space):
    x1=[]
    y1=[]
    h, w = img.shape[0:2]
    for x in range(0,w,space):
        for y in range(0,h,space):
            x1.append(x)
            y1.append(y)
            s=zip(x1,y1)
            anchor_points=list(s)
    #uncomment to display image with anchorpoints
            #cv2.circle(img,(x,y),1,(0,0,0),1)
    return anchor_points

#get anchor boxes based on unitspace
#returns rows and columns of anchor boxes
def get_anchor_boxes(img,unitspace):
    Scales=[1,1.2] # scales
    AspectRatios=[1,1.2] #aspect ratios
    BoxCols_lst=[]
    BoxRows_lst=[]
    for s in Scales:
        for a in AspectRatios:
            BoxCols=s*unitspace
            BoxRows=a*BoxCols
            BoxCols_lst.append(int(BoxCols))
            BoxRows_lst.append(int(BoxRows))
    #print(BoxRows_lst,BoxCols_lst) # (100,200,200,400) (100,100,200,200)

    return BoxCols_lst,BoxRows_lst
