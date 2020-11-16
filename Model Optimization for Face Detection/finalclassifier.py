import numpy as np
import sklearn
import pdb
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid, GroupKFold
import pickle
import cv2
import os
import random
import math
from warnings import simplefilter
from generate_anchorboxes import valid_anchor_points,anchor_points,get_anchor_boxes

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#defining image which is used for testing
img = cv2.imread('../data/images/Writing2Round2Side_frame_0.png',cv2.IMREAD_UNCHANGED)
#img = cv2.imread('../data/images/Pointing1Round2Side_frame_0.png',cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape[0:2]

#defining classifier
class BoxClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,NumOfPCAcomponents,C,gamma):

        self.C_= C
        self.gamma_=gamma
        self.NumOfPCAcomponents_=NumOfPCAcomponents


    def fit(self,X,y):
         self.pca = PCA(n_components=self.NumOfPCAcomponents_, svd_solver='randomized',
                  whiten=True).fit(X)
         X_train_pca = self.pca.transform(X)
         self.clf = svm.SVC(probability=True)
         self.clf.fit(X_train_pca, y)
         return self

    def predict_test(self,X_test):
        X_test_pca = self.pca.transform(X_test)
        y_predict  = self.clf.predict(X_test_pca)
        return y_predict

    def predict_proba(self,X):
        pdb.set_trace()
        X_test_prob=self.pca.transform(X)
        probs=self.clf.predict_proba(X_test_prob)
        return probs

    def predict(self,X_test,y_test):
        X_test_pca = self.pca.transform(X_test)
        y_predict  = self.clf.predict(X_test_pca)
        scores     = accuracy_score(y_test, y_predict)
        conf_matrix = confusion_matrix(y_test, y_predict)
        return scores, conf_matrix

    def score(self,X,y):
        return self.predict(X,y)


def ClassifyImage(img,AnchorPoints,BoxRows,BoxCols):
        Num_FaceBoxes=0
        Num_NonFaceBoxes=0
        face_boxes=[]
        non_face_boxes=[]
        for i in range(0,len(AnchorPoints)-1):
            point=AnchorPoints[i-1]
            (a,b,w,h)=(point[0]-(BoxCols[0]/2),point[1]-(BoxRows[0]/2),point[0]+(BoxCols[0]/2),point[1]+(BoxRows[0]/2))
            crop_img_face=img[int(b):int(h),int(a):int(w)]
            face_path="test_files/arrays/"+str(i) +".npy"
            np.save(face_path,crop_img_face)
            face_img_path="test_files/img/"+str(i) +".png"
            cv2.imwrite(face_img_path,crop_img_face)
        return crop_img_face

def call_classifier(train_files,train_labels,valid_files,valid_labels,test_files,test_labels):
    thisdict = {
                "C": 1000,
                "NumOfPCAcomponents": 10,
                "gamma": 0.001
                }
    outer_scores = []
    test_scores =[]
    clf=BoxClassifier(**thisdict)
    clf.fit(train_files, train_labels)
    # evaluate
    pred=clf.predict(valid_files,valid_labels)
    print("pred_validation:",pred)
    outer_scores.append(clf.score(valid_files,valid_labels))

    pred_test=clf.predict_test(test_files)
    print("pred_test:",pred_test)
    test_scores.append(clf.score(test_files,test_labels))
    print("testscroes:",test_scores)
    print("test_labels:",test_labels)
    print(len(pred_test))
    pdb.set_trace()
    # predict probabilities
    probs = clf.predict_proba(test_files)[:1]
    fpr, tpr, thresholds = roc_curve(test_labels,probs)
    return outer_scores,test_scores,clf

#uncomment to generate numpy arrays for test set without lables
#cf=ClassifyImage(img,valid_anchor_points,BoxRows_lst,BoxCols_lst)

# Datasets
# get all files from directory
f_files   = os.listdir("face/")
random.shuffle(f_files)
nf_files  = os.listdir("noface/")
random.shuffle(nf_files)
f_files_test=os.listdir("test_files/face/")
random.shuffle(f_files_test)
nf_files_test=os.listdir("test_files/noface/")
random.shuffle(nf_files_test)



# Splitting into training and validation
# Right now I am doing 80% Training and 20% validation
# full image for testing

#generating training files
f_train_files   = f_files[0:math.floor(0.8*len(f_files))]
f_train_files   = ["face/" + s for s in f_train_files]
nf_train_files  = nf_files[0:math.floor(0.8*len(nf_files))]
nf_train_files  = ["noface/" + s for s in nf_train_files]
train_files     = f_train_files + nf_train_files
train_labels    = [1]*len(f_train_files) + [0]*len(nf_train_files)

#generating validation files
f_valid_files   = f_files[math.floor(0.8*len(f_files)):-1]
f_valid_files   = ["face/" + s for s in f_valid_files]
nf_valid_files  = nf_files[math.floor(0.8*len(nf_files)):-1]
nf_valid_files   = ["noface/" + s for s in nf_valid_files]
valid_files     = f_valid_files + nf_valid_files
valid_labels    = [1]*len(f_valid_files) + [0]*len(nf_valid_files)

#generating testing files
f_test_files   = f_files_test[0:math.floor(1*len(f_files_test))]
f_test_files   = ["test_files/face/" + s for s in f_files_test]
nf_test_files  = nf_files_test[0:math.floor(1*len(nf_files_test))]
nf_test_files   = ["test_files/noface/" + s for s in nf_test_files]
test_files    = f_test_files + nf_test_files
test_labels    = [1]*len(f_test_files) + [0]*len(nf_test_files)




# Creating labels dictionary
labels = {}
for idx,cur_label in enumerate(train_labels):
	labels[train_files[idx]] = cur_label
for idx,cur_label in enumerate(valid_labels):
	labels[valid_files[idx]] = cur_label
for idx,cur_label in enumerate(test_labels):
	labels[test_files[idx]] = cur_label

#loads data based on naming convention
def load_data(files):
    group_array = []
    for idx,file in enumerate(files):
        group_array = group_array + [int(file.split('_')[1])]
        train=np.load(file)
        if idx == 0:
            train=train.flatten()
            allArrays = train
        else:
            train=train.flatten()
            allArrays=np.vstack((allArrays,train))
    group_array = np.array(group_array)
    return group_array, allArrays

#loads files from test_files
def load_data_test(files):
    for idx,file in enumerate(files):
        #file_path="test_files/arrays/"+files[idx]
        train=np.load(file)
        if idx == 0:
            train=train.flatten()
            allArrays = train
        else:
            train=train.flatten()
            allArrays=np.vstack((allArrays,train))
    return  allArrays

#plotting results(predict values for face) on image
def plot_results(img,AnchorPoints,BoxRows,BoxCols,train_files,train_labels,clf):
    for i in range(0,len(AnchorPoints)-1):
        point=AnchorPoints[i-1]
        (a,b,w,h)=(point[0]-(BoxCols[0]/2),point[1]-(BoxRows[0]/2),point[0]+(BoxCols[0]/2),point[1]+(BoxRows[0]/2))
        crop_img_face=img[int(b):int(h),int(a):int(w)]
        crop_img_face=crop_img_face.flatten()
        pred_test=clf.predict_test(crop_img_face.reshape(1,-1))
        if pred_test==1:
            cv2.rectangle(img,(int(a),int(b)),(int(w),int(h)),(0,0,255),5)
    return img


#Take all anchor points and validating them
#Here these are used only when plotting results
anchor_points= anchor_points(img,100)
BoxCols_lst,BoxRows_lst=get_anchor_boxes(img,170)
valid_anchor_points=valid_anchor_points(img,anchor_points,BoxRows_lst[0],BoxCols_lst[0])

#loading data from files
train_groups, train_files=load_data(train_files)
valid_groups, valid_files=load_data(valid_files)
test_files=load_data_test(test_files)

#calling  classifier

valid_scores,test_scores,clf=call_classifier(train_files,np.array(train_labels),valid_files,
                                            np.array(valid_labels),test_files,np.array(test_labels))
print("validation scores:" ,valid_scores)
print("test scores:",test_scores)
import pdb; pdb.set_trace()
figure=plot_results(img,valid_anchor_points,BoxRows_lst,BoxCols_lst,train_files,train_labels,clf)
cv2.imshow('image',figure)
cv2.waitKey(0)
