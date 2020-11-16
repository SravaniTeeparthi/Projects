import os
import random
import pdb
import math
import pickle
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid, GroupKFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


class BoxClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,NumOfPCAcomponents,C,gamma):

        self.C_= C
        self.gamma_=gamma
        self.NumOfPCAcomponents_=NumOfPCAcomponents


    def fit(self,X,y):
         self.pca = PCA(n_components=self.NumOfPCAcomponents_, svd_solver='randomized',
                  whiten=True).fit(X)
         #self.pca = PCA(n_components=self.NumOfPCAcomponents_, svd_solver='randomized')
         #self.pca.fit(X)
         X_train_pca = self.pca.transform(X)
         self.clf = svm.SVC()
         self.clf.fit(X_train_pca, y)
         return self


    def predict(self,X_test,y_test):
        X_test_pca = self.pca.transform(X_test)
        y_predict  = self.clf.predict(X_test_pca)
        scores     = accuracy_score(y_test, y_predict)
        conf_matrix = confusion_matrix(y_test, y_predict)
        return scores, conf_matrix

    def score(self,X,y):
        return self.predict(X,y)





def nested_cv(X, y,groups, inner_cv, outer_cv, Classifier, parameter_grid,test_files,test_labels):
#def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = []
    for training_samples, test_samples in outer_cv.split(X, y, groups):
        # find best parameter using inner cross-validation
        best_parms = {}
        best_score = -np.inf
        # iterate over parameters
        for parameters in parameter_grid:
            # accumulate score over inner splits
            #print(parameters)
            cv_scores = []
            # iterate over inner cross-validation
            for inner_train, inner_test in inner_cv.split(
                    X[training_samples], y[training_samples],
                    groups[training_samples]):
                # build classifier given parameters and training data
                #pdb.set_trace()
                clf = BoxClassifier(**parameters)
                clf.fit(X[inner_train], y[inner_train])
                # evaluate on inner test set
                score , conf_matrix= clf.score(X[inner_test], y[inner_test])
                cv_scores.append(score)
                #print("processing-inner")
                #print(conf_matrix)
            # compute mean score over inner folds
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                # if better than so far, remember parameters
                best_score = mean_score
                best_params = parameters
        # build classifier on best parameters using outer training set
        #print("best_score:",best_score)
        print("bestparamters:",best_params)
        clf = BoxClassifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])
        # evaluate
        pred=clf.predict(test_files,test_labels)
        #print("pred_test:",pred)
        outer_scores.append(clf.score(X[test_samples], y[test_samples]))
        #print("outerloop")
        #print(outer_scores)
    return np.array(outer_scores)

def load_data(files):
    group_array = []
    for idx,file in enumerate(files):
        group_array = group_array + [int(file.split('_')[3])]
        train=np.load(file)
        if idx == 0:
            train=train.flatten()
            allArrays = train
        else:
            train=train.flatten()
            allArrays=np.vstack((allArrays,train))

        #print(idx)
        #print(np.shape(allArrays))
    #pdb.set_trace()
    group_array = np.array(group_array)
    return group_array, allArrays

# Datasets
# Get all files from faces and no faces
f_files   = os.listdir("face_2_1/")
random.shuffle(f_files)
nf_files  = os.listdir("noface_2_1/")
random.shuffle(nf_files)

# Splitting into training and validation
# Right now I am doing 70% Training and 30% Testing

f_train_files   = f_files[0:math.floor(0.8*len(f_files))]
f_train_files   = ["face_2_1/" + s for s in f_train_files]
nf_train_files  = nf_files[0:math.floor(0.8*len(nf_files))]
nf_train_files  = ["noface_2_1/" + s for s in nf_train_files]
train_files     = f_train_files + nf_train_files
train_labels    = [1]*len(f_train_files) + [0]*len(nf_train_files)

f_test_files   = f_files[math.floor(0.8*len(f_files)):-1]
f_test_files   = ["face_2_1/" + s for s in f_test_files]
nf_test_files  = nf_files[math.floor(0.8*len(nf_files)):-1]
nf_test_files   = ["noface_2_1/" + s for s in nf_test_files]
test_files     = f_test_files + nf_test_files
test_labels    = [1]*len(f_test_files) + [0]*len(nf_test_files)


# Creating labels dictionary
labels = {}
for idx,cur_label in enumerate(train_labels):
	labels[train_files[idx]] = cur_label
for idx,cur_label in enumerate(test_labels):
	labels[test_files[idx]] = cur_label



train_groups, train_files=load_data(train_files)
test_groups,test_files=load_data(test_files)

param_grid = {'C':  [1e3],
            'gamma': [0.0001],
            'NumOfPCAcomponents': [10]}
#param_grid = {'C':  [1e3, 5e3, 1e4, 5e4],
#              'gamma': [0.0001, 0.001, 0.005, 0.01, 0.1],
#             'NumOfPCAcomponents': [10, 50, 100,200]}
print("Parameter grid:\n{}".format(param_grid))


scores=nested_cv(train_files,np.array(train_labels),train_groups, GroupKFold(n_splits=5),
                   GroupKFold(n_splits=5), SVC, ParameterGrid(param_grid),test_files,np.array(test_labels))
print("Cross-validation scores: {}".format(scores))

# Print a list of stats
scores=np.array([scores[0][0],scores[1][0],scores[2][0],scores[3][0],scores[4][0]])
stats_list = (scores.min(), scores.max(), scores.mean())
print("Min value = {:0.2f}, Max value = {:0.2f}, Mean = {:0.2f}".format(*stats_list ))
