from sklearn import ensemble
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from numpy import mean
from sklearn import tree
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn import preprocessing
from sklearn.metrics import classification_report
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
import joblib

warnings.filterwarnings('ignore')

features = np.loadtxt("X.txt")
out = np.loadtxt("Y.txt")
from sklearn.model_selection import KFold
train_x, test_x, train_y, test_y =\
train_test_split(features,out,test_size=0.1)
from xgboost.sklearn import XGBClassifier
print("\033[0;33;40m**********************************************************\033[0m")
print("\033[0;33;40m******AI Platform for Miscarriage Prediction (AI-MP)******\033[0m")
print("\033[0;33;40m**********************************************************\033[0m")

# Cross-validation
print("\033[0;33;40mPlease input the K for K-fold cross-validation:\033[0m")
K = input()
kfold = KFold(n_splits=int(K), shuffle=True, random_state=1)

# Machine learning model
print("\033[0;33;40mPlease choose the machine learning model:\033[0m")
print("\033[0;36;40m1 Extreme Boosting Decision Tree (XGBoost)\033[0m")
print("\033[0;36;40m2 Gradient Boosting Decision Tree (GBDT)\033[0m")
print("\033[0;36;40m3 Random Forest (RF)\033[0m")
print("\033[0;36;40m4 Decision Tree (DT)\033[0m")
Model = input()
# XGBoost
if Model == "1":
    print("\033[0;33;40mPlease set the learning_rate (typically 0.001-0.02):\033[0m")
    lr = input()
    print("\033[0;33;40mPlease set the n_estimators (typically 20-300):\033[0m")
    ne = input()
    print("\033[0;33;40mPlease set the subsample (typically 0.6-1.0):\033[0m")
    ss = input()
    print("\033[0;33;40mPlease set the colsample_bytree (typically 0.6-1.0):\033[0m")
    cb = input()
    print("\033[0;33;40mPlease set the max_depth (typically 2-15):\033[0m")
    depth = input()
    enreg = XGBClassifier(learning_rate=float(lr),n_estimators=int(ne),subsample=float(ss),\
    colsample_bytree=float(cb),max_depth=int(depth),nthread=3)

# GBDT
elif Model == "2":
    print("\033[0;33;40mPlease set the learning_rate (typically 0.001-0.02):\033[0m")
    lr = input()
    print("\033[0;33;40mPlease set the n_estimators (typically 20-300):\033[0m")
    ne = input()
    print("\033[0;33;40mPlease set the subsample (typically 0.6-1.0):\033[0m")
    ss = input()
    print("\033[0;33;40mPlease set the colsample_bytree (typically 0.6-1.0):\033[0m")
    cb = input()
    print("\033[0;33;40mPlease set the max_depth (typically 2-15):\033[0m")
    depth = input()
    enreg = ensemble.GradientBoostingClassifier(n_estimators=int(ne),\
    subsample=float(ss), learning_rate=float(lr),max_depth=int(depth))

# RF
elif Model == "3":
    print("\033[0;33;40mPlease set the n_estimators (typically 20-300):\033[0m")
    ne = input()
    print("\033[0;33;40mPlease set the n_estimators (typically 2-20):\033[0m")
    msl = input()
    enreg = ensemble.RandomForestClassifier(n_estimators=int(ne), n_jobs=2,\
    min_samples_leaf=int(msl),oob_score=True)

elif Model == "4":
    print("\033[0;33;40mPlease choose the kernel function (linear, rbf, sigmoid, poly):\033[0m")
    kernel = input()
    enreg = SVC(kernel=kernel)

scores=[]
auc=[]
total_importance= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(10,11):
    total_importance= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    scores = []
    auc = []
    j = 1
    for train, test in kfold.split(features, out):
        enreg.fit(features[train], out[train])
        # calculate score
        scores.append(enreg.score(features[test], out[test]))
        # calculate auc
        auc.append(roc_auc_score(out[test],enreg.predict(features[test])))
        # save models
        joblib.dump(enreg, "Model-"+str(j)+".pkl")
        j = j + 1
        # save classification report
        cr = classification_report(out[test],enreg.predict(features[test]))
        f = open("classification_report.txt","a+")
        f.write(cr)
        f.close()
        # save importance
        importances = enreg.feature_importances_
        total_importance = total_importance+importances
        for f in range(features.shape[1]):
           fi = open("importance.txt","a+")
           fi.write(str(importances[f]))
           fi.close()
        fi = open("importance.txt","a+")
        fi.write("\n")
        fi.close()
        # calculate roc and auc
        fpr,tpr,thres = roc_curve(out[test],enreg.predict_proba(features[test])[:,1])
        results = np.vstack((fpr, tpr, thres))
        np.savetxt("results"+str(len(scores))+".txt",results.T)
    print("\033[0;33;40m*** Finished !!! ***\033[0m")
    print("\033[0;33;40mAccuracy: \033[0m",scores)
    print("\033[0;33;40mAUC: \033[0m",auc)
    print("\033[0;33;40mImportance: \033[0m",total_importance/10)
    print("\033[0;33;40mClassification report is saved in classification_report.txt\033[0m")
    print("\033[0;33;40mAUCs are saved is result*.txt\033[0m")
    print("\033[0;33;40m*** Finished !!! ***\033[0m")

