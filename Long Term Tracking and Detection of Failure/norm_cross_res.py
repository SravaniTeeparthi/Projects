import cv2
import numpy as np
import sys
import skvideo.io
import pdb
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

if( len(sys.argv) != 2):
    print("USAGE:\n\t python3 play_bb_video.py <method name>\n")
    print("'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'""]")
    sys.exit(1)


template = cv2.imread('../data/templates/G-C3L1P-Feb28-B-Camera2_q2_03-05_2.png',0)
w, h = template.shape[::-1]

cap = cv2.VideoCapture('../data/G-C3L1P-Feb28-B-Camera2_q2_03-05.mp4')

ret, frame = cap.read()
ht, wd, ch = frame.shape
#out = cv2.VideoWriter('out/videos_res/'+ sys.argv[1]+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'),30 , (wd,ht))
#writer = skvideo.io.FFmpegWriter('out/videos_res/'+ sys.argv[1]+'.avi')
Q=100
x=512-Q
#y=93-Q
y=0
a=w+2*Q
b=h+2*Q

bbox_list=[]
fr_to_skip=1
poc=0
max_val_list=[]
while(cap.isOpened() and frame is not None):

    method=sys.argv[1]
    frame1=frame[y:y+a,x:x+b]
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray,template,5)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    #for plotting max values
    max_val_list=max_val_list+[max_val]
    top_left= (top_left[0]+x,top_left[1]+y)
    bottom_right = (top_left[0] + w, top_left[1] + h)

    fr=cv2.rectangle(frame,top_left, bottom_right, (255,255,255), 2)
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

#uncomment to show frame
    cv2.imshow('fr',gray_fr)

#uncomment to write frame
    #writer.writeFrame(gray_fr)
    cv2.waitKey(1)

    #writing as dataframe
    bbox=poc,top_left[0],top_left[1],w,h
    poc  = poc + fr_to_skip
    bbox_list = bbox_list + [bbox]
    ret, frame = cap.read()

cap.release()
#plotting max values for everyframe
plt.plot(max_val_list)
plt.xlabel('frames')
plt.ylabel('coffients')
plt.show()

# Create a dataframe out of list and save it
bbox_df = pd.DataFrame(bbox_list,
                       columns=["poc","x", "y", "w","h"])
bbox_df.to_csv('out/videos_res/csv/'+ sys.argv[1]+ 'Q_100' +".csv",index=False)
#writer.close()
cv2.destroyAllWindows()
