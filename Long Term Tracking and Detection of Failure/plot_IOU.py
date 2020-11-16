import pdb
import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib

ALL_SAMPLES = True

#get csv files
if ALL_SAMPLES:
    matplotlib.rcParams.update({'font.size': 22})
    samp_interval = 30
    plt1=pd.read_csv("python_code/plot/iou_boosting.csv")
    plt1 = list(plt1['IOU'])
    plt1 = plt1[::samp_interval]
    plt2=pd.read_csv("python_code/plot/iou_dcfcsr.csv")
    plt2 = list(plt2['IOU'])
    plt2 = plt2[::samp_interval]
    plt3=pd.read_csv("python_code/plot/iou_kcf.csv")
    plt3 = list(plt3['IOU'])
    plt3 = plt3[::samp_interval]
    plt4=pd.read_csv("python_code/plot/iou_medianflow.csv")
    plt4 = list(plt4['IOU'])
    plt4 = plt4[::samp_interval]
    plt5=pd.read_csv("python_code/plot/iou_mil.csv")
    plt5 = list(plt5['IOU'])
    plt5 = plt5[::samp_interval]
    plt6=pd.read_csv("python_code/plot/iou_mosse.csv")
    plt6 = list(plt6['IOU'])
    plt6 = plt6[::samp_interval]
    plt7=pd.read_csv("python_code/plot/iou_tld.csv")
    plt7 = list(plt7['IOU'])
    plt7 = plt7[::samp_interval]
    plt_nor=pd.read_csv("python_code/plot/iou_cross_correlation.csv")
    plt_nor = list(plt_nor['IOU'])
    plt_nor = plt_nor[::samp_interval]

    fig, ax = plt.subplots()
    x_axis = list(range(0,len(plt1)))
    ax.plot(x_axis,plt1,color='green',label='boosting')
    ax.plot(x_axis,plt2,color='red',label='dcfcsr')
    ax.plot(x_axis,plt3,color='cyan',label='kcf')
    ax.plot(x_axis,plt4,color='magenta',label='medianflow')
    ax.plot(x_axis,plt5,color='yellow',label='mil')
    ax.plot(x_axis,plt6,color='black',label='mosse')
    ax.plot(x_axis,plt7,color='blue',label='tld')
    ax.plot(x_axis, plt_nor,color='orange',label='nxcorr')
    plt.xlabel('1 unit = 1/2 min',fontsize=28)
    plt.ylabel('IOU ratio',fontsize=28)
    plt.title("Background (Trackers) Vs Nx-Cross-Correlation", fontsize=32)

    legend = ax.legend(loc='upper right', shadow=True, fontsize='small')

    plt.show()
else:
    matplotlib.rcParams.update({'font.size': 22})
    plt1=pd.read_csv("python_code/plot/iou_boosting.csv",nrows=100)
    plt2=pd.read_csv("python_code/plot/iou_dcfcsr.csv",nrows=100)
    plt3=pd.read_csv("python_code/plot/iou_kcf.csv",nrows=100)
    plt4=pd.read_csv("python_code/plot/iou_medianflow.csv",nrows=100)
    plt5=pd.read_csv("python_code/plot/iou_mil.csv",nrows=100)
    plt6=pd.read_csv("python_code/plot/iou_mosse.csv",nrows=100)
    plt7=pd.read_csv("python_code/plot/iou_tld.csv",nrows=100)
    plt_nor=pd.read_csv("python_code/plot/iou_cross_correlation.csv",nrows=100)

    fig, ax = plt.subplots()
    ax.plot(plt1,color='green',label='boosting')
    ax.plot(plt2,color='red',label='dcfcsr')
    ax.plot(plt3,color='cyan',label='kcf')
    ax.plot(plt4,color='magenta',label='medianflow')
    ax.plot(plt5,color='yellow',label='mil')
    ax.plot(plt6,color='black',label='mosse')
    ax.plot(plt7,'g--',label='tld')
    ax.plot(plt_nor,color='orange',label='nxcorr')
    plt.xlabel('time (Seconds)',fontsize=28)
    plt.ylabel('IOU ratio',fontsize=28)
    plt.title("Background (Trackers)", fontsize=32)

    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

    plt.show()
