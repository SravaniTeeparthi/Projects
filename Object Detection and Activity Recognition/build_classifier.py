import os
import time
import pdb
import cv2
import numpy as np
from select_best_classifier import SampleDNN,nested_cv,results
from sklearn.model_selection import ParameterGrid, StratifiedKFold

def SingleFB(X_train_ori,y_train_ori,
             X_test,y_test,
             all_batches,all_epochs):
    in_size       = [np.shape(X_train_ori[1])]
    num_first_fil = [2,4]
    num_convnets  = [2,3]
    param_grid = {"input_size": in_size,
                  "num_first_filters": num_first_fil,
                  "num_conv_nets":num_convnets,
                  "structure":"S",
                  "dropout":[0] ,
                  "batch_normalization":"F"}
    train_results = nested_cv(X_train_ori, y_train_ori,   StratifiedKFold(3),
                              StratifiedKFold(3), SampleDNN,
                              ParameterGrid(param_grid),
                              tr_batch_size=all_batches,
                              tr_epochs=all_epochs)
    results(X_train_ori,y_train_ori,train_results,X_test,y_test,all_batches,all_epochs)


def PyramidFB(X_train_ori,y_train_ori,
             X_test,y_test,
             num_first_fil,num_convnets,
             all_batches,all_epochs,):
    """ This function calls the main classifier
     with the structure of pyramid filterbank (4,2) """
    in_size       = [np.shape(X_train_ori[1])]
    param_grid      = {"input_size": in_size,
                        "num_first_filters": num_first_fil,
                        "num_conv_nets":num_convnets,
                        "structure":"P",
                        "dropout":[0],
                        "batch_normalization":"F"}
    train_results   = nested_cv(X_train_ori, y_train_ori,StratifiedKFold(3),
                              StratifiedKFold(3), SampleDNN,
                              ParameterGrid(param_grid),
                              tr_batch_size=all_batches,
                              tr_epochs=all_epochs)
    results(X_train_ori,y_train_ori,train_results,X_test,y_test,all_batches,all_epochs)


def InversePyramidFB(X_train_ori,y_train_ori,
             X_test,y_test,
             num_first_fil,num_convnets,
             all_batches,all_epochs,
             batch_normalization, dropout ):
    """ This function calls the main classifier
     with the structure of pyramid filterbank (4,2) """
    in_size       = [np.shape(X_train_ori[1])]
    param_grid = {"input_size": in_size,
                  "num_first_filters": num_first_fil,
                  "num_conv_nets":num_convnets,
                  "structure":"I",
                  "dropout":dropout,
                  "batch_normalization":batch_normalization}
    train_results = nested_cv(X_train_ori, y_train_ori,StratifiedKFold(3),
                              StratifiedKFold(3), SampleDNN,
                              ParameterGrid(param_grid),
                              tr_batch_size=all_batches,
                              tr_epochs=all_epochs)
    results(X_train_ori,y_train_ori,train_results,X_test,y_test,all_batches,all_epochs)

def OFSingleFB(X_train_ori,y_train_ori,
               X_test,y_test,
              all_batches,all_epochs):
    in_size       = [np.shape(X_train_ori[1])]
    num_first_fil = [2,4]
    num_convnets  = [2,3]
    param_grid = {"input_size": in_size,
                  "num_first_filters": num_first_fil,
                  "num_conv_nets":num_convnets,
                  "structure":"S",
                  "dropout":[0.4] ,
                  "batch_normalization":"T"}
    train_results = nested_cv(X_train_ori, y_train_ori,   StratifiedKFold(3),
                              StratifiedKFold(3), SampleDNN,
                              ParameterGrid(param_grid),
                              tr_batch_size=all_batches,
                              tr_epochs=all_epochs)
    results(X_train_ori,y_train_ori,train_results,X_test,y_test,all_batches,all_epochs)

# X_train_gray    = np.load('./nparrays/tnt/train_50_50/train_gray_X_50.npy')
# y_train_gray    = np.load('./nparrays/tnt/train_50_50/train_gray_y_50.npy')
# X_test          = np.load('./nparrays/tnt/train_50_50/test_gray_X_50.npy')
# y_test          = np.load('./nparrays/tnt/train_50_50/test_gray_y_50.npy')
# print("processing single FB")
# SingleFB(X_train_gray,y_train_gray,X_test,y_test,10,16)
# print("----------------------------------------=-------------------------------------------------------------------")
# print("Processing PyramidFB with N1=8 and 3 layers")
# PyramidFB(X_train_gray,y_train_gray,X_test,y_test,[8],[3],10,16)
# print("-------------------------------------------------------------------------------------------------------------")
# print("Processing PyramidFB with N1=4 and 2 layers")
# PyramidFB(X_train_gray,y_train_gray,X_test,y_test,[4],[2],10,16)
# print("----------------------------------------------------------------------------------------------------------------")
# print("Processing InversePyramidFb with N1=2 and 3 layers with batch normalization and dropout")
# InversePyramidFB(X_train_gray,y_train_gray,X_test,y_test,[2],[3],10,16,"T",[0.4])
# print("--------------------------------------------------------------------------------------------------------------")
# print("Processing InversePyramidFb with N1=4 and 2 layers with batch normalization and dropout")
# InversePyramidFB(X_train_gray,y_train_gray,X_test,y_test,[4],[2],10,16,"T",[0.4])

#optical flow input
X_train_flow     = np.load('./nparrays/tnt/train_50_50/train_flow_X_50.npy')
y_train_flow     = np.load('./nparrays/tnt/train_50_50/train_flow_y_50.npy')
X_test_flow      = np.load('./nparrays/tnt/train_50_50/test_flow_X_50.npy')
y_test_flow      = np.load('./nparrays/tnt/train_50_50/test_flow_y_50.npy')
print("-------------------------------------------------------------------------------------------------------------------")
print("Processing optical flow input with single FB")
OFSingleFB(X_train_flow,y_train_flow,X_test_flow,y_test_flow,10,1)
