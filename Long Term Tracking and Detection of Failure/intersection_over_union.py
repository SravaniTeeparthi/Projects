# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
import os
import sys
import pdb
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


if( len(sys.argv) != 4):
    print("USAGE:\n\t python3 play_bb_video.py <video path> <csv file path> <csv file path_1>\n")
    print("OUTPUT:\n\t Plays video with bounding boxes.")
    print("NOTE:\n\t Do not use spaces and double underscores in object name\n")
    sys.exit(1)

# Readng video file
vr         = cv2.VideoCapture(sys.argv[1])
fr_suc, fr = vr.read()
ht, wd, ch = fr.shape
if not(fr_suc):
    print("ERROR:\n\t Unable to read video file,")
    print("\t",sys.argv[1])
    sys.exit(1)

# Reading csv file
df = pd.read_csv(sys.argv[2])
#reading second csv file
df_1 = pd.read_csv(sys.argv[3])
poc_1=df_1['poc']

font = cv2.FONT_HERSHEY_SIMPLEX


head, tail = os.path.split(sys.argv[2])
tail=tail.split(".")[0]
alg_name=tail.split("__")[1]
out = cv2.VideoWriter('out/IOU/'+ tail+'_Q_100'+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'),30 , (wd,ht))



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea =max(0,(xB - xA + 1)) * max(0,(yB - yA + 1))

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    if iou>=1:
        iou=1
    return round(iou,2)

I_ratio_lst=[]
# data frame loop
for idx,row in df.iterrows():
    poc     = row['poc']
    bbox    = (row['x'], row['y'] ,
               row['w'], row['h'])
    #flag    = row['flag']
    try:
        df_1row = df_1[df_1['poc'] == poc]
        bbox_1=(int(df_1row['x']),
                int(df_1row['y']),
                int(df_1row['w']),
                int(df_1row['h']))
    except:
        pdb.set_trace()
    # set video to poc
    vr.set(cv2.CAP_PROP_POS_FRAMES,poc)
    fr_suc, fr = vr.read()
    # Drawing bounding box
    fr_bb   = cv2.rectangle(fr, bbox, (0,0,255), 3)#red -gt
    fr_bb_1=cv2.rectangle(fr_bb,bbox_1, (0,255,0), 3) #green -method

    #calculation of imtersection over uniuon
    bbox_I=(bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])
    bbox_1_I=(bbox_1[0],bbox_1[1],bbox_1[0]+bbox_1[2],bbox_1[1]+bbox_1[3])
    I_ratio=bb_intersection_over_union(bbox_I,bbox_1_I)
    #if I_ratio>=0.5:
    #	cv2.putText(fr_bb_1,"success"+format(I_ratio),(30,50), font,2,(255,128,0),2,cv2.LINE_AA)
    #else:
        #cv2.putText(fr_bb_1,"failure"+format(I_ratio),(30,50), font,2,(255,128,0),2,cv2.LINE_AA)

    # Show frame with bounding box
    #cv2.imshow("Tracking",fr_bb_1)
    #pdb.set_trace()
    #out.write(fr_bb_1);
    #cv2.waitKey(6)
    I_ratio_lst = I_ratio_lst + [I_ratio]

vr.release()
out.release()
plt.plot(I_ratio_lst)
plt.xlabel('frames')
plt.ylabel('IOU ratio')
#plt.show()
# Create a dataframe out of list and save it
bbox_df = pd.DataFrame(I_ratio_lst,
                       columns=["IOU"])
bbox_df.to_csv('plot/'+ "iou_nxcorr_with_failure_handling_02-05" ".csv",index=False)
#writer.close
cv2.destroyAllWindows()
