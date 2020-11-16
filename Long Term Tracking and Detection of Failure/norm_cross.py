import cv2
import numpy as np
import sys
import skvideo.io
import pdb
from matplotlib import pyplot as plt
from PIL import Image

if( len(sys.argv) != 2):
    print("USAGE:\n\t python3 play_bb_video.py <method name>\n")
    print("'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'""]")
    sys.exit(1)


template = cv2.imread('C:/Users/Sravani/Desktop/Project-533/cross_correlation/img/G-C3L1P-Feb28-B-Camera2_q2_03-05_2.png',0)
w, h = template.shape[::-1]

cap = cv2.VideoCapture('C:/Users/Sravani/Desktop/Project-533/sravani/G-C3L1P-Feb28-B-Camera2_q2_03-05.mp4')

ret, frame = cap.read()
ht, wd, ch = frame.shape
out = cv2.VideoWriter('out/videos/'+ sys.argv[1]+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'),30 , (wd,ht))
writer = skvideo.io.FFmpegWriter('out/videos/'+ sys.argv[1]+'.avi')

while(cap.isOpened() and frame is not None):

    method=sys.argv[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray,template,eval(method))

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    fr=cv2.rectangle(gray,top_left, bottom_right,(255,255,255), 2)
    #import pdb; pdb.set_trace()
    #cv2.imshow('fr',fr)

    writer.writeFrame(fr)
    #cv2.waitKey(1)
    ret, frame = cap.read()

cap.release()
writer.close()
cv2.destroyAllWindows()
