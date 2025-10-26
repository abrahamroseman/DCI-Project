#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ============================================================
# TrackingAlgorithms_DataLoading_Class
# ============================================================

#Libraries
import os
import h5py 

class TrackingAlgorithms_DataLoading_Class:
    """
    A utility class for saving and loading Tracking Algorithm results
    """

    @staticmethod
    def SaveData(ModelData,DataManager, Dictionary, timeString): 
        """
        Save tracking algorithm results to an HDF5 file.
        """
        
        fileName = f"{DataManager.dataName}_{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz_{timeString}.h5"
        filePath = os.path.join(DataManager.outputDataDirectory,fileName)
        
    
        with h5py.File(filePath, 'w') as f:
            for varName, varData in Dictionary.items():
                f.create_dataset(f"{varName}", data=varData, compression="gzip")
    
        print(f"Saved output to {filePath}","\n")

    @staticmethod
    def LoadData(ModelData, DataManager, timeString):
        """
        Load tracking algorithm results from an HDF5 file.
        """
        fileName = f"{DataManager.dataName}_{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz_{timeString}.h5"
        filePath = os.path.join(DataManager.outputDataDirectory, fileName)

        if not os.path.exists(filePath):
            raise FileNotFoundError(f"HDF5 file not found:\n{filePath}")

        Dictionary = {}
        with h5py.File(filePath, 'r') as f:
            for key in f.keys():
                Dictionary[key] = f[key][:]
        
        print(f"Loaded data from {filePath} ({len(Dictionary)} variables)\n")
        return Dictionary

# #HOW TO LOAD
# #IMPORT CLASSES
# sys.path.append(os.path.join(mainCodeDirectory,"3_Project_Algorithms","2_Tracking_Algorithms"))
# from CLASSES_TrackingAlgorithms import TrackingAlgorithms_DataLoading_Class
        
# #EXAMPLE USAGE
# TrackingAlgorithms_DataLoading_Class.SaveData(ModelData,DataManager, Dictionary, timeString)
# Dictionary = TrackingAlgorithms_DataLoading_Class.LoadData(ModelData,DataManager, timeString)

