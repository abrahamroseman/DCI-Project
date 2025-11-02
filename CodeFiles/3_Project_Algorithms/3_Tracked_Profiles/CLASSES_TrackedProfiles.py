#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ============================================================
# DomainProfiles_DataLoading_CLASS
# ============================================================

#Libraries
import os
import pickle

class TrackedProfiles_DataLoading_CLASS:
    """
    A utility class for saving and loading tracked profile results
    """

    @staticmethod
    def SaveProfile(ModelData,DataManager, Dictionary, dataName, t):
        profileType = "TrackedProfiles"
        timeString = t if isinstance(t, str) else ModelData.timeStrings[t]
        
        fileName = f"{profileType}_{dataName}_{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz_{timeString}.pkl"
        filePath = os.path.join(DataManager.outputDataDirectory,fileName)
        
        with open(filePath, "wb") as f:
            pickle.dump(Dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        print(f"Saved output to {filePath}","\n")

    @staticmethod
    def LoadProfile(ModelData, DataManager, dataName, t):
        """
        Load a saved TrackedProfiles .pkl file for a given time index t.
        """
        profileType = "TrackedProfiles"
        timeString = t if isinstance(t, str) else ModelData.timeStrings[t]
        
        fileName = f"{profileType}_{dataName}_{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz_{timeString}.pkl"
        filePath = os.path.join(DataManager.outputDataDirectory,fileName)
    
        with open(filePath, "rb") as f:
            Dictionary = pickle.load(f)
    
        # print(f"Loaded profile dictionary from {filePath}\n")
        return Dictionary


#Example Call
#IMPORT CLASSES
# sys.path.append(os.path.join(mainCodeDirectory,"3_Project_Algorithms","3_Tracked_Profiles"))
# from CLASSES_TrackedProfiles import TrackedProfiles_DataLoading_CLASS

#Example Run
#saving
# TrackedProfiles_DataLoading_CLASS.SaveProfile(ModelData,DataManager_TrackedProfiles, profileArraysDictionary, t)
#loading
# TrackedProfiles_DataLoading_CLASS.LoadProfile(ModelData,DataManager_TrackedProfiles, t)

