from scipy.signal import correlate2d
import numpy as np
import pandas as pd
import pdb
import sys
import cv2

TRAIN_FLAG = True
# Read csv file
if TRAIN_FLAG:
    dffull = pd.read_csv("gt_Vs_nxcorr_train.csv")
else:
    dffull = pd.read_csv("gt_Vs_nxcorr_test.csv")
gtbbox = dffull.iloc[0]
dfs    = dffull[dffull['label'] == 's']
dff    = dffull[dffull['label'] == 'f']


# dfs = dfs.sample(n=len(dff)) # --> Switch on for training.
df  = pd.concat([dff,dfs])
df  = df.reset_index()

# Reading columns
pocs = np.array(df['poc'])
xgt  = np.array(df['xgt'])
ygt  = np.array(df['ygt'])
wgt  = np.array(df['wgt'])
hgt  = np.array(df['hgt'])
x    = np.array(df['x'])
y    = np.array(df['y'])
w    = np.array(df['w'])
h    = np.array(df['h'])

# Video loop
if TRAIN_FLAG:
    vidh        = cv2.VideoCapture('../../data/G-C3L1P-Feb28-B-Camera2_q2_03-05.mp4')
else:
    vidh        = cv2.VideoCapture('../../data/G-C3L1P-Feb28-B-Camera2_q2_02-05.mp4')
        

# Getting template from GT (POC = 0)
vidh.set(cv2.CAP_PROP_POS_FRAMES,0)
fr_suc, fr = vidh.read()
if not(fr_suc):
    print("ERROR: cannot read template frame")
gfr = cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
tmp      = gfr[gtbbox['ygt'] : gtbbox['ygt'] + gtbbox['hgt'],
                gtbbox['xgt'] : gtbbox['xgt'] + gtbbox['wgt']]
tmphist = cv2.calcHist([tmp],[0],None,[20],[0,256])

sidx = 0
fidx = 0
for idx,row in df.iterrows():

    # POC
    poc = row['poc']

    # Seek video to current POC
    vidh.set(cv2.CAP_PROP_POS_FRAMES,poc)
    fr_suc, fr = vidh.read()
    if not(fr_suc):
        print("ERROR: cannot read template frame")
        gfr = cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)

    # Tracker Image
    tbbox    = df[ df['poc'] == poc ]
    trimg    = gfr[ int(tbbox['y']) : int(tbbox['y']) + int(tbbox['h']),
                     int(tbbox['x']) : int(tbbox['x']) + int(tbbox['w']) ]

    # Tracker image Y histogram
    trhist   = cv2.calcHist([trimg],[0],None,[20],[0,256])

    # Histogram difference
    diff_hist = tmphist - trhist

    # Save histograms as np arrays in appropriate folders
    try:
        l = tbbox.label.iloc[0]
        if (l == 's'):
            if TRAIN_FLAG:
                np.save("./hists_train/s/"+ str(sidx)+".npy",diff_hist)
            else:
                np.save("./hists_test/s/"+ str(sidx)+".npy",diff_hist)
            sidx = sidx + 1
        else:
            if TRAIN_FLAG:
                np.save("./hists_train/f/"+ str(fidx)+".npy",diff_hist)
            else:
                np.save("./hists_test/f/"+ str(fidx)+".npy",diff_hist)
            fidx = fidx + 1
    except:
        pdb.set_trace()

vidh.release()
