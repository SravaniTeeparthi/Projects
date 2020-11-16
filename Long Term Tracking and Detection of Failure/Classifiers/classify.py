import os
import glob
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# Training
# Reading all success
for idx,filepath in enumerate(glob.iglob("./hists_train/s/*.npy")):
    if idx == 0:
        sarr = np.load(filepath)
    else:
        sarr = np.concatenate((sarr,np.load(filepath)), axis=1)
sarr = sarr.transpose()
sdf  = pd.DataFrame(sarr)
sdf['l'] = 1

# Reading all failed cases
for idx,filepath in enumerate(glob.iglob("./hists_train/f/*.npy")):
    if idx == 0:
        farr = np.load(filepath)
    else:
        farr = np.concatenate((farr,np.load(filepath)), axis=1)
farr = farr.transpose()
fdf  = pd.DataFrame(farr)
fdf['l'] = 0

# Combining both dataframes
dffull = pd.concat([sdf,fdf])

# Training
train       = dffull
#train, test = train_test_split(dffull, test_size=0.3)

# Creating X and y for training
y = np.array(train['l'])
traincopy = train.copy()
X = np.array(traincopy.drop(['l'],axis=1))

n_samples, n_features = X.shape
cv = StratifiedKFold(n_splits=10)
classifier = RandomForestClassifier(n_estimators=30, max_depth=13,
                             random_state=0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=24)
plt.ylabel('True Positive Rate',fontsize=24)
plt.title('ROC and AUC for Random forest',fontsize=24)
plt.legend(loc="lower right",fontsize='xx-large')
#plt.show()


# xgboost
model = XGBClassifier()
model.fit(X, y)















# Testing
for idx,filepath in enumerate(glob.iglob("./hists_test/s/*.npy")):
    if idx == 0:
        sarr = np.load(filepath)
    else:
        sarr = np.concatenate((sarr,np.load(filepath)), axis=1)
sarr = sarr.transpose()
sdf  = pd.DataFrame(sarr)
sdf['l'] = 1

# Reading all failed cases
for idx,filepath in enumerate(glob.iglob("./hists_test/f/*.npy")):
    if idx == 0:
        farr = np.load(filepath)
    else:
        farr = np.concatenate((farr,np.load(filepath)), axis=1)
farr = farr.transpose()
fdf  = pd.DataFrame(farr)
fdf['l'] = 0

# Combining both dataframes
dffull = pd.concat([sdf,fdf])

# Testing
test       = dffull
#train, test = train_test_split(dffull, test_size=0.3)

# Creating X and y for testing
y = np.array(test['l'])
testcopy = test.copy()
X_test = np.array(testcopy.drop(['l'],axis=1))

# Testing and printing stats
y_pred = classifier.predict(X_test) # rm
y_pred_xg = model.predict(X_test) #xgboost



# Simple failure handling
"""
On failure we will update current bounding box to be average of previous 21
successful bounding boxes.
"""
# Reading testing coordinates csv

df_full = pd.read_csv('gt_Vs_nxcorr_test.csv')
df_copy = df_full.as_matrix(columns=['x','y','w','h'])
df_gt   = df_full.as_matrix(columns=['xgt','ygt','wgt','hgt'])
poc_arr = df_full.as_matrix(columns=['poc'])
# Dropping ground truth rows, flag, POC and index
for i,lab in enumerate(y_pred):
    if not(lab): # if the classifier gives failure
        j  = i - 1
        num_bbox = 1
        while(j >= 0 and num_bbox < 30):
            if y_pred[j] == 1: # store bounding boxes for only success cases
                if num_bbox == 1:
                    bbox_lst = [[df_full.iloc[j]['x'], df_full.iloc[j]['y'],
                                 df_full.iloc[j]['w'], df_full.iloc[j]['h']]]
                else:
                    bbox_lst = bbox_lst +\
                               [[df_full.iloc[j]['x'], df_full.iloc[j]['y'],
                                 df_full.iloc[j]['w'], df_full.iloc[j]['h']]]
            j = j - 1
            num_bbox = num_bbox + 1
        bbox_arr = np.array(bbox_lst)
        print("-----------")
        print(df_copy[i])
        print(df_gt[i])
        bbox_new_coords = np.rint(np.mean(bbox_arr, axis=0))
        print(bbox_new_coords)
        print("-----")
        df_copy[i] = bbox_new_coords
# Adding POC
df_copy = np.hstack((poc_arr, df_copy))
df_fail_handled = pd.DataFrame(df_copy, columns=['poc','x','y','w','h'])
df_fail_handled.to_csv("Simple_failure_handling_02-05.csv",index=False)



# cnf matrix
cnf = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)
print("========= Random forest =========")
print(cnf)
print(acc)
print("========= XGBoost =========")
cnf = confusion_matrix(y, y_pred_xg)
acc = accuracy_score(y, y_pred_xg)
print(cnf)



plt.show()
