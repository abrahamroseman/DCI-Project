#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
    def LoadData(ModelData, DataManager, timeString,
                 dataName=None,outputDataDirectory=None,
                 printstatement=True):
        """
        Load tracking algorithm results from an HDF5 file.
        """
        if dataName is None:
            dataName = DataManager.dataName
        if outputDataDirectory is None:
            outputDataDirectory = DataManager.outputDataDirectory
        
        fileName = f"{dataName}_{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz_{timeString}.h5"
        filePath = os.path.join(outputDataDirectory, fileName)

        if not os.path.exists(filePath):
            raise FileNotFoundError(f"HDF5 file not found:\n{filePath}")

        Dictionary = {}
        with h5py.File(filePath, 'r') as f:
            for key in f.keys():
                Dictionary[key] = f[key][:]

        if printstatement == True:
            print(f"Loaded data from {filePath} ({len(Dictionary)} variables)\n")
        return Dictionary

# #HOW TO LOAD
# #IMPORT CLASSES
# sys.path.append(os.path.join(mainCodeDirectory,"3_Project_Algorithms","2_Tracking_Algorithms"))
# from CLASSES_TrackingAlgorithms import TrackingAlgorithms_DataLoading_Class
        
# #EXAMPLE USAGE
# TrackingAlgorithms_DataLoading_Class.SaveData(ModelData,DataManager, Dictionary, timeString)
# Dictionary = TrackingAlgorithms_DataLoading_Class.LoadData(ModelData,DataManager, timeString)


# In[ ]:


# ============================================================
# SlurmJobArray_Class
# ============================================================

class SlurmJobArray_Class:
    
    @staticmethod
    def StartSlurmJobArray(num_jobs,num_slurm_jobs, ISRUN):
        job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) #this is the current SBATCH job id
        if job_id==0: job_id=1
        if ISRUN==False:
            start_job=1;end_job=num_jobs+1
            return start_job,end_job
        total_elements=num_jobs #total num of variables
    
        job_range = total_elements // num_slurm_jobs  # Base size for each chunk
        remaining = total_elements % num_slurm_jobs   # Number of chunks with 1 extra 
        
        # Function to compute the start and end for each job_id
        def get_job_range(job_id, num_slurm_jobs):
            job_id-=1
            # Add one extra element to the first 'remaining' chunks
            start_job = job_id * job_range + min(job_id, remaining)
            end_job = start_job + job_range + (1 if job_id < remaining else 0)
        
            if job_id == num_slurm_jobs - 1: 
                end_job = total_elements 
            return start_job, end_job
        # def job_testing():
        #     #TESTING
        #     start=[];end=[]
        #     for job_id in range(1,num_slurm_jobs+1):
        #         start_job, end_job = get_job_range(job_id)
        #         print(start_job,end_job)
        #         start.append(start_job)
        #         end.append(end_job)
        #     print(np.all(start!=end))
        #     print(len(np.unique(start))==len(start))
        #     print(len(np.unique(end))==len(end))
        # job_testing()
        # if sbatch==True:
            
        start_job, end_job = get_job_range(job_id, num_slurm_jobs)
        index_adjust=start_job
        # print(f'start_job = {start_job}, end_job = {end_job}')
        if start_job==0: start_job=1
        if end_job==total_elements: end_job+=1
        return start_job,end_job
    
    @staticmethod
    def StartJobArray(ModelData, job_id,num_jobs):
        total_elements=ModelData.Np #total num of variables
    
        if num_jobs >= total_elements:
            raise ValueError("Number of jobs cannot be greater than or equal to total elements.")
        
        job_range = total_elements // num_jobs  # Base size for each chunk
        remaining = total_elements % num_jobs   # Number of chunks with 1 extra 
        
        # Function to compute the start and end for each job_id
        def get_job_range(job_id, num_jobs):
            job_id-=1
            # Add one extra element to the first 'remaining' chunks
            start_job = job_id * job_range + min(job_id, remaining)
            end_job = start_job + job_range + (1 if job_id < remaining else 0)
        
            if job_id == num_jobs - 1: 
                end_job = total_elements #- 1
            return start_job, end_job
        # def job_testing():
        #     #TESTING
        #     start=[];end=[]
        #     for job_id in range(1,num_jobs+1):
        #         start_job, end_job = get_job_range(job_id)
        #         print(start_job,end_job)
        #         start.append(start_job)
        #         end.append(end_job)
        #     print(np.all(start!=end))
        #     print(len(np.unique(start))==len(start))
        #     print(len(np.unique(end))==len(end))
        # job_testing()
    
        # if sbatch==True:
        #     job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) #this is the current SBATCH job id
        #     if job_id==0: job_id=1
            
        start_job, end_job = get_job_range(job_id, num_jobs)
        index_adjust=start_job
        # print(f'start_job = {start_job}, end_job = {end_job}')
        return start_job,end_job,index_adjust
    
    @staticmethod
    def job_filter(arr, start_job,end_job):
        return arr[(arr[:,0]>=start_job)&(arr[:,0]<end_job)]
    
    @staticmethod
    def ApplyJobArray_Nested(trackedArrays, start_job, end_job):
        """
        Apply job-array filtering to all arrays inside the nested trackedArrays dictionary
        """
    
        trackedArrays_filtered = {}
    
        for main_key, sub_dict in trackedArrays.items():
            trackedArrays_filtered[main_key] = {}
    
            for sub_key, arr in sub_dict.items():
                # Apply job filtering
                filteredArray = job_filter(arr, start_job, end_job)
                trackedArrays_filtered[main_key][sub_key] = filteredArray
    
        print(f"Completed job filter for {len(trackedArrays_filtered)} main categories ({start_job} → {end_job})")
        return trackedArrays_filtered


# In[ ]:


# ============================================================
# Results_InputOutput_Class
# ============================================================

import os
import h5py

class Results_InputOutput_Class:
    """
    A static utility class for saving and loading tracking algorithm results.
    """

    @staticmethod
    def SaveOutFile(ModelData,DataManager, Dictionary,job_id): 
        """
        Save tracking algorithm results to an HDF5 file.
        """
        
        fileName = f"{DataManager.dataName}_{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz_job{job_id}.h5"
        filePath = os.path.join(DataManager.outputDataDirectory,fileName)
        
    
        with h5py.File(filePath, 'w') as f:
            for varName, varData in Dictionary.items():
                f.create_dataset(f"{varName}", data=varData, compression="gzip")
    
        print(f"Saved output to {filePath}","\n")

    @staticmethod
    def LoadOutFile(ModelData, DataManager, job_id, varName=None, printstatement=False): 
        """
        Load tracking algorithm results from an HDF5 file and return as a dictionary.
        """
    
        fileName = f"{DataManager.dataName}_{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz_job{job_id}.h5"
        filePath = os.path.join(DataManager.outputDataDirectory, fileName)
    
        if printstatement==True:
            print(f"Loading output from {filePath}\n")
    
        Dictionary = {}
        with h5py.File(filePath, 'r') as f:
            if varName is None:
                # Load all variables
                Dictionary = {name: f[name][:] for name in f.keys()}
                return Dictionary
            else:
                if varName not in f:
                    raise KeyError(f"{varName} not found in {filePath}")
                arr = f[varName][:]
                return arr

    @staticmethod
    def SaveAllCloudBase_Job(ModelData,DataManager,
                             all_cloudbase,job_id):
        Dictionary = {"all_cloudbase": all_cloudbase}
        Results_InputOutput_Class.SaveOutFile(ModelData,DataManager, Dictionary,f"{job_id}_all_cloudbase")

    @staticmethod
    def SaveAllCloudBase_Combined(ModelData,DataManager,
                                  all_cloudbase):
        Dictionary = {"all_cloudbase": all_cloudbase}
        Results_InputOutput_Class.SaveOutFile(ModelData,DataManager, Dictionary,f"combined_all_cloudbase")

    @staticmethod
    def LoadAllCloudBase_Job(ModelData,DataManager,
                             job_id):
        out = Results_InputOutput_Class.LoadOutFile(ModelData,DataManager,f"{job_id}_all_cloudbase")
    
        return out

    @staticmethod
    def LoadAllCloudBase_Combined(ModelData,DataManager):
        out = Results_InputOutput_Class.LoadOutFile(ModelData,DataManager,f"combined_all_cloudbase")
        return out

    #######################################################
    
    @staticmethod
    def SaveLFC_Profile_Job(ModelData,DataManager,
                            LFC_profile,job_id, Ltype):
        #Ltype in LFC or LCL
        Dictionary = {f"{Ltype}_profile": LFC_profile} 
        Results_InputOutput_Class.SaveOutFile(ModelData,DataManager, Dictionary,f"{job_id}_{Ltype}_profile")

    @staticmethod
    def SaveLFC_Profile_Combined(ModelData,DataManager,
                            LFC_profile, Ltype):
        #Ltype in LFC or LCL
        Dictionary = {f"{Ltype}_profile": LFC_profile} 
        Results_InputOutput_Class.SaveOutFile(ModelData,DataManager, Dictionary,f"combined_{Ltype}_profile")

    @staticmethod
    def LoadLFC_Profile_Job(ModelData,DataManager,
                            job_id, Ltype):
        #Ltype in LFC or LCL
        out = Results_InputOutput_Class.LoadOutFile(ModelData,DataManager,f"{job_id}_{Ltype}_profile")
    
        return out

    @staticmethod
    def LoadLFC_Profile_Combined(ModelData,DataManager, Ltype):
        #Ltype in LFC or LCL
        out = Results_InputOutput_Class.LoadOutFile(ModelData,DataManager,f"combined_{Ltype}_profile")
    
        return out


# In[1]:


# ============================================================
# TrackedParcel_Loading_Class
# ============================================================

import numpy as np

class TrackedParcel_Loading_Class:

    @staticmethod
    def LoadFinalData(ModelData,DataManager,Results_InputOutput_Class):    
        Dictionary = Results_InputOutput_Class.LoadOutFile(ModelData,DataManager,job_id="combined_SUBSET")
        return Dictionary

    @staticmethod
    def GetTrackedParcelArrays(Dictionary):
        """
        Extract all tracked parcel arrays (CL, nonCL, SBF, ColdPool) from Dictionary.
    
        Returns
        -------
        dict
            Nested dictionary of arrays, grouped by type and category.
        """
        trackedArrays = {
            "CL": {
                "ALL": Dictionary["CL_ALL_out_arr"],
                "SHALLOW": Dictionary["CL_SHALLOW_out_arr"],
                "DEEP": Dictionary["CL_DEEP_out_arr"],
            },
            "nonCL": {
                "ALL": Dictionary["nonCL_ALL_out_arr"],
                "SHALLOW": Dictionary["nonCL_SHALLOW_out_arr"],
                "DEEP": Dictionary["nonCL_DEEP_out_arr"],
            },
            "SBF": {
                "ALL": Dictionary["SBF_ALL_out_arr"],
                "SHALLOW": Dictionary["SBF_SHALLOW_out_arr"],
                "DEEP": Dictionary["SBF_DEEP_out_arr"],
            },
            "nonSBF": {
                "ALL": Dictionary["nonSBF_ALL_out_arr"],
                "SHALLOW": Dictionary["nonSBF_SHALLOW_out_arr"],
                "DEEP": Dictionary["nonSBF_DEEP_out_arr"],
            },
            "ColdPool": {
                "ALL": Dictionary["ColdPool_ALL_out_arr"],
                "SHALLOW": Dictionary["ColdPool_SHALLOW_out_arr"],
                "DEEP": Dictionary["ColdPool_DEEP_out_arr"],
            }
        }
    
        # concise summary
        print(f"CL: ALL={len(trackedArrays['CL']['ALL'])}, SHALLOW={len(trackedArrays['CL']['SHALLOW'])}, DEEP={len(trackedArrays['CL']['DEEP'])}")
        print(f"nonCL: ALL={len(trackedArrays['nonCL']['ALL'])}, SHALLOW={len(trackedArrays['nonCL']['SHALLOW'])}, DEEP={len(trackedArrays['nonCL']['DEEP'])}")
        print(f"SBF: ALL={len(trackedArrays['SBF']['ALL'])}, SHALLOW={len(trackedArrays['SBF']['SHALLOW'])}, DEEP={len(trackedArrays['SBF']['DEEP'])}")
        print(f"ColdPool: ALL={len(trackedArrays['ColdPool']['ALL'])}, SHALLOW={len(trackedArrays['ColdPool']['SHALLOW'])}, DEEP={len(trackedArrays['ColdPool']['DEEP'])}")
    
        return trackedArrays
    
    
    #Reading In Final Results from SubsetParcels
    @staticmethod
    def LoadingSubsetParcelData(ModelData,DataManager,Results_InputOutput_Class):
    
        #Loading Tracked Parcel Data
        Dictionary = TrackedParcel_Loading_Class.LoadFinalData(ModelData,DataManager,Results_InputOutput_Class)
        trackedArrays = TrackedParcel_Loading_Class.GetTrackedParcelArrays(Dictionary)
        
        #cloudbase
        all_cloudbase = Results_InputOutput_Class.LoadAllCloudBase_Combined(ModelData,DataManager)["all_cloudbase"]
    
        mean_all_cloudbase = np.nanmean(all_cloudbase)
        min_all_cloudbase = np.nanmin(all_cloudbase)
        print(f"Mean Cloudbase is: {mean_all_cloudbase:.2f} km\n")
        print(f"Min Cloudbase is: {min_all_cloudbase:.2f} km\n")
    
        #lfc and lcl
        LFC_profile = Results_InputOutput_Class.LoadLFC_Profile_Combined(ModelData,DataManager,Ltype='LFC')["LFC_profile"]
        LCL_profile = Results_InputOutput_Class.LoadLFC_Profile_Combined(ModelData,DataManager,Ltype='LCL')["LCL_profile"]
        
        #LFC and LCL
        MeanLFC=np.mean(LFC_profile)
        MeanLCL=np.mean(LCL_profile)
        MinLFC=np.min(LFC_profile)
        MinLCL=np.min(LCL_profile)
        print(f"Mean LFC is: {MeanLFC:.2f} km\n")
        print(f"Mean LCL is: {MeanLCL:.2f} km\n")
        print(f"Min LFC is: {MinLFC:.2f} km\n")
        print(f"Min LCL is: {MinLCL:.2f} km\n")
        
    
        #combining all level data into dictionary
        LevelsDictionary = {"all_cloudbase": all_cloudbase,
                            "mean_all_cloudbase": mean_all_cloudbase,
                            "min_all_cloudbase": min_all_cloudbase,
    
                            "LFC_profile": LFC_profile,
                            "LCL_profile": LCL_profile,
                            "MeanLFC": MeanLFC,
                            "MeanLCL": MeanLCL,
                            "MinLFC": MinLFC,
                            "MinLCL": MinLCL}
                            
                            
        return trackedArrays,LevelsDictionary

# #Example Calls
# trackedArrays,LevelsDictionary = TrackedParcel_Loading_Class.LoadingSubsetParcelData(ModelData,DataManager,
#                                                          Results_InputOutput_Class)
    
# CL_ALL = trackedArrays["CL"]["ALL"]
# SBF_DEEP = trackedArrays["SBF"]["DEEP"]
# ColdPool_SHALLOW = trackedArrays["ColdPool"]["SHALLOW"]

