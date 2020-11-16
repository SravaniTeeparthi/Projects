# read csv file
# get that particulr images and coordinates
# flip it
#save changes cordinates and names with _flip
# make coding as simple as pssobile
# use arguemtns after done

import pandas as pd
import pdb
import numpy as np
import cv2
from data_aug.data_aug import *
from data_aug.bbox_util import *
import matplotlib.pyplot as plt
import pickle as pkl


gt_csv = pd.read_csv("../frames/csv_files/aolmegt_hands_original.csv")
image_path = "C:/Softwares/ImageAugmentation/frames/original/"
len = len(gt_csv)
save_image_path = "C:/Softwares/ImageAugmentation/frames/Augmented_flip/"
df_flip = pd.DataFrame()
for i in range(0,len):
    image_name = gt_csv.iloc[i].filename
    img = cv2.imread(image_path + image_name)[:,:,::-1]
    xmin = np.float(gt_csv.iloc[i].xmin)
    ymin = np.float(gt_csv.iloc[i].ymin)
    xmax = np.float(gt_csv.iloc[i].xmax)
    ymax = np.float(gt_csv.iloc[i].ymax)
    bboxes=np.array([[xmin, ymin, xmax, ymax]])
    plotted_img = draw_rect(img, bboxes)
    plt.imshow(plotted_img)
    plt.show()
    img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
    #print(bboxes_)
    plotted_img = draw_rect(img_, bboxes_)
    plt.imshow(plotted_img)
    plt.show()
    filename = gt_csv.iloc[i].filename
    filename = filename.split(".")[0] + "_flip.png"
    fp_xmin = int(bboxes_[0][0])
    fp_ymin = int(bboxes_[0][1])
    fp_xmax = int(bboxes_[0][2])
    fp_ymax = int(bboxes_[0][3])
    width = 858
    height = 480
    row = pd.DataFrame({"filename" : filename,
                        "width" : width,
                        "height" : height,
                        "class" : "hand",
                        "xmin" : fp_xmin,
                        "ymin" : fp_ymin,
                        "xmax" : fp_xmax,
                        "ymax" : fp_ymax} , index=[0])
    #pdb.set_trace()
    df_flip = df_flip.append(row)
    print(i)

    image_name = save_image_path + filename
    #cv2.imwrite(image_name, img_)
print(df_flip)
#df_flip.to_csv("flip_csv.csv", index = False)
