#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 20:06:20 2020

@author: shurookalmohamadi
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-



from sklearn.model_selection import LeaveOneOut, RepeatedKFold

import tsfel
import numpy as np
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sn
from sklearn.metrics import roc_curve
from pycm import ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support
import unittest
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import calibration_curve
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier

# from pomegranate import *


from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
import xlwt
from sklearn.model_selection import cross_val_score

filename = 'glove_AB_TEFSL'
# filename = 'glove_TEFSL_LFilter'
# ExcelFile= 'CA_glove_1'

# window profile
# filename='glove_TEFSL_nofilter'
# filename='glove_TEFSL_win750_LFilter'
filename='glove_TEFSL_win750'
# filename='glove_TEFSL_W750_nofilter'

# ExcelFile= 'CA_glove_window'




dataset1 = pd.read_csv(filename+".csv")

def fill_missing_values(df):
    #Handle eventual missing data. Strategy: replace with mean.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df
dataset = fill_missing_values(dataset1)
print(filename)
NoFeatures=dataset.shape
# print('Features vector Shape',NoFeatures)


#
corr_features = tsfel.correlated_features(dataset) #Highly correlated features are removed
dataset.drop(corr_features, axis=1, inplace=True)
print('Features vector Shape after removing Highly correlated features',dataset.shape)
# print(dataset)

# np.save('Selected_data/'+filename +'.csv', dataset)
dataset.to_csv('Selected_data/'+filename+'.csv', index=False)
dataset.head()
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print (y_train.shape)
#
selector = VarianceThreshold() # Remove low variance features
X = selector.fit_transform(X)
selectedFeatures=X.shape
print('Features vector Shape after removing VarianceThreshold',selectedFeatures)
# print('--------------------------------------------------\n\n')
#

# Feature Scaling
Scaler = StandardScaler()
X = Scaler.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=4)
# print(y_val.shape)
# print(X_train.shape)

# -------------------------------------------------
#print('\nRandom Forest Classifier:\n')


models = [ RandomForestClassifier(n_estimators= 1000, min_samples_split= 5, min_samples_leaf=1, max_features='sqrt', max_depth= 100, bootstrap= False,random_state=0,criterion='entropy')]
# models = [LogisticRegression(solver='liblinear',C=1, penalty='l2', max_iter=1000)]
# models = [LogisticRegression()]
# models = [KNeighborsClassifier(n_neighbors=1, metric='euclidean')]
# models = [SVC(C=100,gamma=0.01,kernel='rbf',probability=True),KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
# LogisticRegression(solver='liblinear',C=1, penalty='l2', max_iter=1000),LinearDiscriminantAnalysis()]
# models = LinearDiscriminantAnalysis(),
# models = ExtraTreesClassifier(n_estimators=n_estimators),
# models =  GaussianNB()

i=5
randonCV=[15]

for model in models:
    print('--------------------------------------------------')
    print(model)
    print('--------------------------------------------------')
    for randomState in randonCV :
        print('random_state',randomState)
        cv = StratifiedKFold(n_splits=5,random_state=randomState,shuffle=True)
        loo = LeaveOneOut()
        Accuracy = []
        precision = []
        recall = []
        F1_score = []
        FNR = []
        TPR = []
        TNR = []
        FPR = []
        ERR_mean=[]
        mean_squaredError = []
        for train, test in cv.split(X, y):
            userID = y[test]
            model.fit(X[train], y[train])


            pred_test = model.predict(X[test])
            proba_test = model.predict_proba((X[test]))

            df_proba_test = pd.DataFrame(proba_test)
            # print('proba_test', df_proba_test)
            # print(classification_report(y[test], pred_test))
            cm = ConfusionMatrix(y[test], pred_test)
            # print(cm)
            df_proba_test = pd.DataFrame(proba_test)

            idx = 0
            df_proba_test.insert(loc=idx, column='User', value=userID)
            df_confusion_matrix = pd.DataFrame(confusion_matrix(y[test], pred_test))

            # with pd.ExcelWriter(ExcelFile + '.xlsx', mode='a') as writer:
            #     df_proba_test.to_excel(writer, index=False, header=True, sheet_name=filename)
            #     df_confusion_matrix.to_excel(writer, index=False, header=True, sheet_name=filename + 'M')

            # print('mean_squared_error',mean_squared_error(y[test],pred_test))
            # print('\nRF Overall Accuracy=', round(cm.Overall_ACC, 3))
            precision_recall_fscore = precision_recall_fscore_support(y[test], pred_test, average='weighted',labels=np.unique(pred_test))
            # print('\nprecision', round(precision_recall_fscore[0], 2))
            # print('recall', round(precision_recall_fscore[1], 2))
            # print('F1_score', round(precision_recall_fscore[2], 2))
            # print('\nFNR Micro =', round(cm.FNR_Micro, 3))
            #         # print('TPR Micro =', round(cm.TPR_Micro, 3))
            #         # print('TPR  =', cm.TPR)
            #         # print('\nTNR Micro =', round(cm.TNR_Micro, 3))
            #         # print('FPR Micro =', round(cm.FPR_Micro, 3))
            #print(classification_report(y[test], pred_test))
            ERR = 0
            # print('EER_All',cm.ERR)

            for val in cm.ERR.values():
                ERR += val
            ERR = ERR / len(cm.ERR)
            # print('EER',round(ERR, 3))
            Accuracy.append(round(cm.Overall_ACC, 3))
            mean_squaredError.append(mean_squared_error(y[test], pred_test))
            precision.append(round(precision_recall_fscore[0], 3))
            recall.append(round(precision_recall_fscore[1], 3))
            F1_score.append(round(precision_recall_fscore[2], 3))
            TPR.append(round(cm.TPR_Micro, 3))
            FNR.append(round(cm.FNR_Micro, 3))
            TNR.append(round(cm.TNR_Micro, 3))
            FPR.append(round(cm.FPR_Micro, 3))
            ERR_mean.append(round(ERR, 3))
            # print('\nERR =', cm.ERR)
            # data = list(cm.ERR.items())
            # an_array = np.array(data)
            # # print('\nERR =',an_array)
            # ERR = an_array[:, 1].mean()
            # # print('\nERR =', round(ERR, 4))
            # EER_array = an_array[:, 1]
            # # print('\nERR =', EER_array[userID])
            # #
            # # data = list(cm.TPR.items())
            # # an_array1 = np.array(data)
            # # TPR_array = an_array1[:, 1]
            # # print('TPR=', an_array1[:, 1])
            # #
            # data = list(cm.FPR.items())
            # an_array2 = np.array(data)
            # FPR_array = an_array2[:, 1]
            # # print('FPR=', FPR_array)
            # # print('\nERR =', EER_array[i-1], '\tFPR=', FPR_array[i-1])


            # print(' ----------------------------------------')


        data = {'fold':  [1,2,3,4,5],
        'F1_score': F1_score,
        'TPR':TPR,
        'FPR':FPR,
        'FNR':FNR,
        'TNR':TNR,
        'mean_squared_Error':mean_squaredError,
        'Accuracy': Accuracy,
        'ERR_mean':ERR_mean
        }
        info={'info':['DataSet','Features Type ','#Features','Features selection','#selected','seed','models'],
              'infoV':['glove(13 users/15 tasks)',filename,NoFeatures,'Remove Highly correlated features and Remove low variance features',selectedFeatures,randomState,models]}


        df = pd.DataFrame (data, columns = ['fold','Accuracy','F1_score','TPR','FPR','FNR','TNR','ERR_mean'])
        df2=pd.DataFrame (info, columns = ['info','infoV'])
        print (df)

        # print('precision', np.mean(precision))
        # print('recall',np.mean(recall))
        # print('F1_score', np.mean(F1_score))
        # print('ERR_mean', np.mean(ERR_mean))
        # print('Accuracy', np.mean(Accuracy))

        print('\n-----------------------------')
        print('F1_score', np.mean(F1_score) * 100)
        print('TPR', np.mean(TPR) * 100)
        print('FPR', np.mean(FPR) * 100)
        print('EER', np.mean(ERR_mean) * 100)
        print('-----------------------------')

        book = load_workbook('TSEFL_glove.xlsx')
        writer = pd.ExcelWriter('TSEFL_glove.xlsx', engine='openpyxl')
        writer.book = book

        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

        df.to_excel(writer, filename,index = False, header=True)
        df2.to_excel(writer, filename,index = False, header=False, startrow=len(df)+2)

        writer.save()



if __name__ == '__main__':
    unittest.main()
