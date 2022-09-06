import csv
from random import randint, choice,uniform
import matplotlib.pyplot as plt
import pandas as pd
import tsfel
import numpy as np
from numpy import concatenate
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn import svm
# from sklearn.neural_network import MLPClassifier
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
from scipy import signal
from scipy.signal import savgol_filter
import tsfel

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier

# from pomegranate import *


from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
import xlwt
from sklearn.model_selection import cross_val_score



# window profile

#
# def retreiveSingleUserData(userID):
#     x = np.genfromtxt('data/singleUserX' + str(userID) + '.csv', delimiter=",", skip_header=False)
#     y = np.genfromtxt('data/singleUserY' + str(userID) + '.csv', delimiter=",", skip_header=False)
#     m = x.shape[0]
#     return x, y.reshape((m, 1))
#
# def loadSingleData(filename):
#     data = np.loadtxt(open(filename+".csv",'rb'), delimiter=',', skiprows=1)
#     # print('loadSingleData',data.shape)
#     labels = data[:, 0]
#     data_points = data[:, 1:]
#     # print('labels',labels)
#     return labels, data_points

# def singleUserData(userID,file):
#     filename = file
#     labels, data_points = loadSingleData(filename)
#
#     # print(labels)
#     x=[]
#     y=[]
#     print('labels',labels)
#     for i in range(len(labels)):
#         # print('labels[i]=',labels[i],'userID',userID)
#         if labels[i]==userID:
#             x.append(data_points[i])
#             y.append(labels[i])
#     # print('y=====', y)
#     np.savetxt('data/singleUserX' + str(userID) + '.csv', x, delimiter=',')
#     np.savetxt('data/singleUserY' + str(userID) + '.csv', y, delimiter=',')


class Input_Attcker:

    def ImportData(self, DataFile):
        print('filename is  = ', DataFile)
        Data = pd.read_csv((DataFile), skiprows=[1])
        # UserData = np.delete(Data, 0, axis=1)
        Data = np.array(Data).astype(float)
        return Data

    def target_raw(self,user):
        fileID = 'Participant_' + str(user) + '/Participant_' + str(user) + '_Trial_' + str(U) + '.csv'
        Data = pd.read_csv((fileID), skiprows=[1])
        raw= Data.shape[0]
        return raw

    def filterData(self,rawdata):
            b, a = signal.butter(4, 0.03, analog=False)
            imp_lf = signal.lfilter(b, a, rawdata)
            return imp_lf

    def Creatprofile(self, user, Trail,
                     thumb_force_Feature, index_force_Feature,
                     middle_force_Feature, palm_force_Feature,
                     thumb_angle_Feature, index_angle_Feature, middle_angle_Feature):
        profile = []

        for n in range(len(thumb_force_Feature)):
            window = n + 1
            profile = []
            thumbF_W1 = thumb_force_Feature.iloc[n, 0:].values
            indexF_W1 = index_force_Feature.iloc[n, 0:].values
            middleF_W1 = middle_force_Feature.iloc[n, 0:].values

            palmF_W1 = palm_force_Feature.iloc[n, 0:].values
            thumbA_W1 = thumb_angle_Feature.iloc[n, 0:].values
            indexA_W1 = index_angle_Feature.iloc[n, 0:].values
            middleA_W1 = middle_angle_Feature.iloc[n, 0:].values

            profile.append(user)
            profile.append(window)
            # profile.append(Task_Seq)
            profile.append(Trail)
            # profile.append(T)

            for w in thumbF_W1:
                # print ('w',window,w)
                profile.append(w)
            for w in indexF_W1:
                # print ('w',window,w)
                profile.append(w)
            for w in middleF_W1:
                # print ('w',window,w)
                profile.append(w)

            for w in palmF_W1:
                # print ('w',window,w)
                profile.append(w)
            for w in thumbA_W1:
                # print ('w',window,w)
                profile.append(w)
            for w in indexA_W1:
                # print ('w',window,w)
                profile.append(w)
            for w in middleA_W1:
                # print ('w',window,w)
                profile.append(w)

            # print('\n----------------------------------------------------------------------------')
            print('profile', profile)
            # print('\n----------------------------------------------------------------------------')
            # Feature_1.ExportFeatures('glove_B_TEFSL_statistical' ,profile)
            Attack.ExportFeatures('forgeid_Sample_450_3best/Attack_sample_seg_'+str(target_user), profile)

    def Create_profile(self, target_user, Trail, thumb_force, index_force, middle_force, palm_force, thumb_angle,
                       index_angle, middle_angle):

        thumb_force_Feature = Attack.FindFeatures(thumb_force)
        index_force_Feature = Attack.FindFeatures(index_force)
        middle_force_Feature = Attack.FindFeatures(middle_force)
        palm_force_Feature = Attack.FindFeatures(palm_force)

        thumb_angle_Feature = Attack.FindFeatures(thumb_angle)
        index_angle_Feature = Attack.FindFeatures(index_angle)
        middle_angle_Feature = Attack.FindFeatures(middle_angle)

        Attack.Creatprofile(target_user,Trail,
                               thumb_force_Feature, index_force_Feature,
                               middle_force_Feature, palm_force_Feature,
                               thumb_angle_Feature, index_angle_Feature, middle_angle_Feature)

    #

    def profile(self, Data):

        Data = pd.DataFrame(list(Data))

        thumb_force = Data.iloc[:, 0].values
        index_force = Data.iloc[:, 1].values
        middle_force = Data.iloc[:, 2].values
        palm_force = Data.iloc[:, 3].values

        thumb_angle = Data.iloc[:, 4].values
        index_angle = Data.iloc[:, 5].values
        middle_angle = Data.iloc[:, 6].values


        thumb_force = np.array(thumb_force).astype(float)
        index_force = np.array(index_force).astype(float)
        middle_force = np.array(middle_force).astype(float)
        palm_force = np.array(palm_force).astype(float)

        thumb_angle = np.array(thumb_angle).astype(float)
        index_angle = np.array(index_angle).astype(float)
        middle_angle = np.array(middle_angle).astype(float)

        # filter Data

        thumb_force = Attack.filterData(thumb_force)
        index_force = Attack.filterData(index_force)
        middle_force = Attack.filterData(middle_force)
        palm_force = Attack.filterData(palm_force)

        thumb_angle = Attack.filterData(thumb_angle)
        index_angle = Attack.filterData(index_angle)
        middle_angle = Attack.filterData(middle_angle)

        Attack.Create_profile(target_user, Trail, thumb_force, index_force, middle_force, palm_force, thumb_angle,
                       index_angle, middle_angle)
        # return  thumb_force, index_force, middle_force, palm_force, thumb_angle, index_angle, middle_angle

    def FindFeatures(self, Tool_featuer):
        # statistical, temporal,spectral
        cfg = tsfel.get_features_by_domain()
        # print (cfg)
        # Extract features

        X = tsfel.feature_extraction.features.zero_cross(Tool_featuer)
        # X = tsfel.time_series_features_extractor(cfg, Tool_featuer)
        X = tsfel.time_series_features_extractor(cfg, Tool_featuer,
                                                 fs=50,window_size=750)  # Receives a time series sampled at 50 Hz, divides into windows of size 250 (i.e. 5 seconds) and extracts all features
        # for key, value in X.items():
        #     print(key, ' : ', value)
        print(Tool_featuer.__len__())
        print(X)
        return X

    def ExportFeatures (self,filename,Data):
            print ('********************************************************')
            print ('Export Features to CSV file (Features_File ) '+filename)
            print('********************************************************')

            with open(filename +'.csv', 'a') as writeFile:
                writer = csv.writer(writeFile , delimiter = ',')
                writer.writerow(Data)
            writeFile.close()

    def Create_forgeid_Sample(self,AttackSample):
        NoFakeSample = self.target_raw(target_user)  # of generated forged samples
        print('NoFakeSample',NoFakeSample)
        binNumber = 3  # of best bins
        bin = 450
        NoFeatures=AttackSample.shape[1]
        Attackdata = []
        for forgeid in range(1, NoFeatures ):
            temp = []
            # print('feature # ',forgeid)
            feature = AttackSample.iloc[:, forgeid].values



            # print('feature=', feature.shape)
            freq, edges = np.histogram(feature, bins=bin)
            # print('befor sort freq', freq)
            # print('edges', edges)

            Sortedfreq = -np.sort(-freq)
            # print('freq', Sortedfreq)
            index = np.argsort(freq)
            # print('index', index)
            freq_index = index[::-1]
            # print('freq_index',freq_index)
            Best_index = freq_index[:binNumber]
            print('Best_index', Best_index)
            for i in Best_index:
                print('index=', i, 'freq=', freq[i])
                print('index+1=', i + 1, 'freq=', freq[i + 1])

                print('edges=', edges[i], 'edges+1=', edges[i + 1])
                edgesSize = edges[i + 1] - edges[i]
                print('E=', edgesSize)

                print('end edges', edges[i] + edgesSize)
                # print(edges[i+1] - edges[i + 2])

                temp.append(edges[i])
                # temp.append(edges[i+1])

            Attackdata.append(temp)
            print('edges most populated', Attackdata)

            # print('freq_index',freq_index)
            # print('Best_index',Best_index)
            # print('\n----------------------------------------')
        print('Attackdata=', Attackdata)

        FakeSample = []
        for n in range(0, NoFakeSample):
            attack_Input = []
            # attack_Input.append(0)
            for i in Attackdata:
                # print('i=',i)
                # for _ in range(1):
                #     print(choice([uniform(i[0], i[0]+edgesSize), uniform(i[1], i[1]+edgesSize),
                #                   uniform(i[2], i[2]+edgesSize)]))

                RandomValue = choice([uniform(i[0], i[0] + edgesSize), uniform(i[1], i[1] + edgesSize),
                                      uniform(i[2], i[2] + edgesSize)])

                # v = np.random.choice(i, 1)[0]
                print('rang=[', i[0], ',', i[0] + edgesSize, ']U[', i[1], ',', i[1] + edgesSize, ']U[', i[2], ',',
                      i[2] + edgesSize, ']')
                print('RandomValue=', RandomValue)

                attack_Input.append(RandomValue)
                # attack_Input.append(v)

            # print(attack_Input.__len__())
            FakeSample.append(attack_Input)
        # print('FakeSample')
        FakeLable=np.zeros(NoFakeSample)
        # FakeLable = np.ones(NoFakeSample) * target_user
        win_Size = 750
        if FakeSample.__len__() > win_Size:
             Attack.profile(FakeSample)
        # print('FakeLable',FakeLable)
        # np.savetxt('forgeid_Sample_50_3best/Fake_ToAttackX_seg' + str(target_user) + '.csv', FakeSample, delimiter=',')
        # np.savetxt('forgeid_Sample_50_3best/Fake_ToAttackY_seg' + str(target_user) + '.csv', FakeLable, delimiter=',')


#===========================[   Main   ]=====================
target_user= 0
Trail = 0
if __name__ == '__main__':
    taregt=[5535, 541, 909,3327, 5521, 8410, 2193,
              2274, 5124, 5319, 8524, 9266, 9875]
    # taregt = [ 3327]
    for t in taregt:
        target_user=t
        print('target_user ID:', target_user)
        print('----------------------------------')

        userID = [5535, 541, 909, 3327, 5521, 8410, 2193,
                  2274, 5124, 5319, 8524, 9266, 9875]
        # userID = [541, 909]

        userID.remove(target_user)
        # userID = ['541', '909', '3327', '5521', '5535', '8410', '2193', '2274', '5124', '5319', '8524', '9266', '9875']

        for U in range(1, 16):
            GroupData=[]
            for user in userID:
                print('User ID:', user)
                print('----------------------------------')

                Setup = 'AB'
                S = 0

                fileID = 'Participant_' + str(user) + '/Participant_' + str(user) + '_Trial_' + str(U) + '.csv'
                Task_Seq = S
                Trail = U
                Time = []

                ''' Creat object '''
                Attack  = Input_Attcker()
                AllData = Attack.ImportData(fileID)
                # print('Number of raw= ', AllData)

                for i in AllData:
                    GroupData.append(i)

                print('Number of raw GroupData= ', GroupData.__len__())
            G = pd.DataFrame(list(GroupData))
            print(' GroupData= ', G.shape)

            # for i in range (3):
            Attack.Create_forgeid_Sample(G)


