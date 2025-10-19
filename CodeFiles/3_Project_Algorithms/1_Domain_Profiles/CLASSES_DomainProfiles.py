#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# DomainProfiles_CLASS
# ============================================================

#Libraries
import numpy as np

class DomainProfiles_Class:
    """
    A utility class for constructing and binning vertical domain profiles.
    """
    
    # @staticmethod
    # def GetCloudyUpdraftThresholds(): #not needed since eulerian_binary arrays were calculated previously
    #     w_thresh1=0.1
    #     w_thresh2=0.5
    #     qcqi_thresh=1e-6
    #     return w_thresh1,w_thresh2,qcqi_thresh

    @staticmethod
    def InitializeProfiles(VARs, ModelData):
        zhs = ModelData.zh
        
        # Initialize profiles for each variable
        profiles = {}
        for var in VARs:
            profiles[var] = np.zeros((len(zhs), 3))  # column 1: var, column 2: counter, column 3: list of zhs
            profiles[var][:, 2] = zhs
    
        #####
        VARs_squares = [key + "_squares" for key in VARs.keys()]
        for var_squares in VARs_squares:
            profiles[var_squares] = np.zeros((len(zhs), 3))  # column 1: var, column 2: counter, column 3: list of zhs
            profiles[var_squares][:, 2] = zhs
        #####
    
        return profiles, VARs_squares

    @staticmethod
    def GetUpdraftMask(data_type, A_g, A_c):
        # Threshold mask
        if data_type == "general":
            where_updraft = (A_g==True)
        elif data_type == "cloudy":
            where_updraft = (A_c==True)
        return where_updraft

    @staticmethod
    def GetIndexes(where_updraft):
        # Get Indexes
        z_ind, y_ind, x_ind = np.where(where_updraft)
        return z_ind, y_ind, x_ind

    @staticmethod
    def MakeProfiles(VARs, VARs_squares, profiles, where_updraft, z_ind):
        # Make Profiles
        # Iterate over each variable in var_names and bin the data
        for (var,var_squares) in zip(VARs,VARs_squares):
            masked_data = VARs[var][where_updraft]
            np.add.at(profiles[var][:, 0], z_ind, masked_data)
            np.add.at(profiles[var][:, 1], z_ind, 1)
            np.add.at(profiles[var_squares][:, 0], z_ind, masked_data**2)
            np.add.at(profiles[var_squares][:, 1], z_ind, 1)
        return profiles

    @staticmethod
    def RegularAverage(VARs, VARs_squares, profiles):
        """
        Compute vertical profiles for the full domain (no masking).
        Uses horizontal means over (y, x) for each vertical level.
        """
        for var, var_squares in zip(VARs, VARs_squares):
            var_data = VARs[var]
    
            # Mean and squared mean profiles
            mean_profile = np.mean(var_data, axis=(1, 2))
            mean_sq_profile = np.mean(var_data**2, axis=(1, 2))
    
            # Store results
            profiles[var][:, 0] = mean_profile
            profiles[var][:, 1] = 1  # dummy counter (all points counted)
            profiles[var_squares][:, 0] = mean_sq_profile
            profiles[var_squares][:, 1] = 1
        return profiles

    @staticmethod
    def DomainProfile(VARs,data_type, A_g,A_c, ModelData, masked=True):
    
        # Initialize profiles for each variable
        profiles, VARs_squares = DomainProfiles_Class.InitializeProfiles(VARs, ModelData)
    
        # Threshold mask
        if masked == True:
            where_updraft = DomainProfiles_Class.GetUpdraftMask(data_type, A_g, A_c)
        elif masked == False:
            # where_updraft = (A_c == False) | (A_c == True)
            profile = DomainProfiles_Class.RegularAverage(VARs, VARs_squares, profiles)
            return profiles
    
        # Get Indexes
        z_ind, _, _ = DomainProfiles_Class.GetIndexes(where_updraft)
    
        # Make Profiles
        profiles = DomainProfiles_Class.MakeProfiles(VARs, VARs_squares, profiles, where_updraft, z_ind)
    
        return profiles


# #Example Call
# #IMPORT CLASSES
# sys.path.append(os.path.join(mainCodeDirectory,"3_Project_Algorithms","1_Domain_Profiles"))
# from CLASSES_DomainProfiles import DomainProfiles_Class

# #Example Run
# Dictionary = DomainProfiles_Class.DomainProfile(VARs, 'general', A_g, A_c, masked=False)


# In[ ]:


# ============================================================
# DomainProfiles_DataLoading_CLASS
# ============================================================

#Libraries
import os
import h5py 

class DomainProfiles_DataLoading_Class:
    """
    A utility class for saving and loading domain profile results
    """
    
    @staticmethod
    def GetCloudyUpdraftThresholds(): #not needed since eulerian_binary arrays were calculated previously
        w_thresh1=0.1
        w_thresh2=0.5
        qcqi_thresh=1e-6
        return w_thresh1,w_thresh2,qcqi_thresh

    @staticmethod
    def SaveProfile(ModelData,DataManager, Dictionary, dataName, datatype, timeString, masked): 
        profileType = "UpdraftProfiles" if masked else "DomainProfiles"
        
        fileName = f"{dataName}_{profileType}_{datatype}_{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz_{timeString}.h5"
        filePath = os.path.join(DataManager.outputDataDirectory,fileName)
        
    
        with h5py.File(filePath, 'w') as f:
            for varName, varProfile in Dictionary.items():
                f.create_dataset(f"{varName}_{datatype}_{timeString}", data=varProfile, compression="gzip")
    
        print(f"Saved output to {filePath}","\n")

    @staticmethod
    def LoadProfile(ModelData,DataManager, dataName, datatype, timeString, masked, printstatement=False):
        profileType = "UpdraftProfiles" if masked else "DomainProfiles"
        
        fileName = f"{dataName}_{profileType}_{datatype}_{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz_{timeString}.h5"
        filePath = os.path.join(DataManager.outputDataDirectory, "..", dataName, fileName)
    
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"Profile file not found: {filePath}")
    
        Dictionary = {}
        with h5py.File(filePath, 'r') as f:
            for key in f.keys():
                Dictionary[key] = f[key][:]
    
        if printstatement==True:
            print(f"Loaded profile data from {filePath}\n")
        return Dictionary


# #Example Call
# #IMPORT CLASSES
# sys.path.append(os.path.join(mainCodeDirectory,"3_Project_Algorithms","1_Domain_Profiles"))
# from CLASSES_DomainProfiles import DomainProfiles_DataLoading_CLASS

# #Example Run
# Dictionary = DomainProfiles_DataLoading_CLASS.SaveProfile(ModelData,DataManager, Dictionary, dataName, datatype=datatype, timeString=timeString, masked=True)

# Dictionary = LoadProfile(ModelData,DataManager, dataName, datatype, timeString,masked,printstatement=False)

