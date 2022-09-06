#!/usr/bin/env python

#  Stage 2 Extracting Featurs 

# Jun 2019 Shurook Amohamade
# The university of sheffield   http://www.sheffield.ac.uk/
#==================================================================================================================================
# This script:  
#
#==================================================================================================================================

'''Importing required libraries '''

#import scipy.spatial.distance as Dist

import csv
import pandas as pd
import cmath
import numpy as np
from cmath import rect, phase
from math import radians, degrees
from scipy import signal
import tsfel

'''---------------------------------------------------- '''

#====================================================================================================================

class Features:
    
    def filterData(self,t,rawdata):
            b, a = signal.butter(4, 0.03, analog=False)
            imp_ff = signal.filtfilt(b, a, rawdata)
            imp_lf = signal.lfilter(b, a, rawdata)
            return imp_lf
            
        
    

    def ImportData(self,DataFile):

            print ('filename is  = ', DataFile)
            
            Data = pd.read_csv((DataFile), skiprows=[1])
 
            time_stamp=Data.iloc[:, 0].values

            thumb_force=Data.iloc[:, 1].values
            index_force=Data.iloc[:, 2].values
            middle_force=Data.iloc[:, 3].values
            palm_force=Data.iloc[:, 4].values
        
            thumb_angle=Data.iloc[:, 5].values
            index_angle=Data.iloc[:, 6].values
            middle_angle=Data.iloc[:, 7].values


            time_stamp=np.array(time_stamp).astype(float)
            thumb_force=np.array(thumb_force).astype(float)
            index_force=np.array(index_force).astype(float)
            middle_force=np.array(middle_force).astype(float)
            palm_force=np.array(palm_force).astype(float)
            
            thumb_angle=np.array(thumb_angle).astype(float)
            index_angle=np.array(index_angle).astype(float)
            middle_angle=np.array(middle_angle).astype(float)
            
            # filter Data
            

            # thumb_force = Features.filterData(self,time_stamp, thumb_force)
            # index_force = Features.filterData(self,time_stamp, index_force)
            # middle_force = Features.filterData(self,time_stamp, middle_force)
            # palm_force = Features.filterData(self,time_stamp, palm_force)
            #
            # thumb_angle = Features.filterData(self,time_stamp, thumb_angle)
            # index_angle = Features.filterData(self,time_stamp, index_angle)
            # middle_angle = Features.filterData(self,time_stamp, middle_angle)



            return Data, time_stamp,thumb_force,index_force,middle_force,palm_force,thumb_angle,index_angle,middle_angle




    def FindFeatures(self,Tool_featuer):
        # statistical, temporal,spectral
        cfg = tsfel.get_features_by_domain()

        # Extract features
        X = tsfel.feature_extraction.features.zero_cross(Tool_featuer)
        # X = tsfel.time_series_features_extractor(cfg, Tool_featuer)
        # Receives a time series sampled at 50 Hz, divides into windows of size 750 (i.e. 15 seconds) and extracts all features
        X = tsfel.time_series_features_extractor(cfg, Tool_featuer, fs=50,window_size=750)
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



    def Creatprofile(self,user,Task_Seq ,Trail,T,
                                           thumb_force_Feature,index_force_Feature,
                                           middle_force_Feature,palm_force_Feature,
                                           thumb_angle_Feature,index_angle_Feature,middle_angle_Feature):
            profile =[]

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
                Feature_1.ExportFeatures('glove_TEFSL_W750_nofilter', profile)

    def Create_profile(self,user,Task_Seq ,Trail,T,thumb_force,index_force,middle_force,palm_force,thumb_angle,index_angle,middle_angle):
   

                    thumb_force_Feature= Feature_1.FindFeatures(thumb_force)
                    index_force_Feature= Feature_1.FindFeatures(index_force)
                    middle_force_Feature= Feature_1.FindFeatures(middle_force)
                    palm_force_Feature= Feature_1.FindFeatures(palm_force)

                    thumb_angle_Feature= Feature_1.FindFeatures(thumb_angle)
                    index_angle_Feature= Feature_1.FindFeatures(index_angle)
                    middle_angle_Feature= Feature_1.FindFeatures(middle_angle)

                    Feature_1.Creatprofile(user,Task_Seq ,Trail,T,
                                           thumb_force_Feature,index_force_Feature,
                                           middle_force_Feature,palm_force_Feature,
                                           thumb_angle_Feature,index_angle_Feature,middle_angle_Feature)
#

#===========================[   Main   ]=====================

if __name__ == '__main__':
    userID = ['541','909','3327','5521','5535','8410','2193','2274','5124','5319','8524','9266','9875']
    for user in userID:
        print('User ID:',user)
        print('----------------------------------')
        for U in range (1,16):
           Setup='AB'
           S=0

           fileID='Participant_'+user+'/Participant_'+user+'_Trial_'+str(U)+'.csv'
           Task_Seq=S
           Trail=U
           Time=[]
        
           ''' Creat object '''
           Feature_1 = Features()
           AllData, time_stamp,thumb_force,index_force,middle_force,palm_force,thumb_angle,index_angle,middle_angle= Feature_1.ImportData(fileID)
           #print('thumb_force',thumb_force)
           T= time_stamp[len(time_stamp)-1]-time_stamp[0]
           print('Number of raw= ',time_stamp.__len__())
           win_Size=1250
           if time_stamp.__len__()>win_Size:
               Feature_1.Create_profile(user,Task_Seq ,Trail,T,thumb_force,index_force,middle_force,palm_force,thumb_angle,index_angle,middle_angle)
           # for t in range(len(time_stamp)):
           #     Time.append(time_stamp[t]-time_stamp[0])

        

    
