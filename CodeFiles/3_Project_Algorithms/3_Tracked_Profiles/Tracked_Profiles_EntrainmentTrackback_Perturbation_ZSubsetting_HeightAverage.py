#!/usr/bin/env python
# coding: utf-8

# In[100]:


####################################
#ENVIRONMENT SETUP


# In[101]:


#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import xarray as xr

import sys; import os; import time; from datetime import timedelta
import pickle
import h5py
from tqdm import tqdm


# In[102]:


#MAIN DIRECTORIES
def GetDirectories():
    mainDirectory='/mnt/lustre/koa/koastore/torri_group/air_directory/Projects/DCI-Project/'
    mainCodeDirectory=os.path.join(mainDirectory,"Code/CodeFiles/")
    scratchDirectory='/mnt/lustre/koa/scratch/air673/'
    codeDirectory=os.getcwd()
    return mainDirectory,mainCodeDirectory,scratchDirectory,codeDirectory

[mainDirectory,mainCodeDirectory,scratchDirectory,codeDirectory] = GetDirectories()


# In[103]:


#IMPORT CLASSES
sys.path.append(os.path.join(mainCodeDirectory,"2_Variable_Calculation"))
from CLASSES_Variable_Calculation import ModelData_Class, SlurmJobArray_Class, DataManager_Class


# In[104]:


#IMPORT FUNCTIONS
sys.path.append(os.path.join(mainCodeDirectory,"2_Variable_Calculation"))
import FUNCTIONS_Variable_Calculation
from FUNCTIONS_Variable_Calculation import *


# In[106]:


#data loading class
ModelData = ModelData_Class(mainDirectory, scratchDirectory, simulationNumber=1)
#data manager class
DataManager = DataManager_Class(mainDirectory, scratchDirectory, ModelData.res, ModelData.t_res, ModelData.Nz_str,
                                ModelData.Np_str, dataType="Tracking_Algorithms", dataName="Lagrangian_UpdraftTracking",
                                dtype='float32',codeSection = "Project_Algorithms")


# In[107]:


#data manager class (for saving data)
DataManager_TrackedProfiles = DataManager_Class(mainDirectory, scratchDirectory, ModelData.res, ModelData.t_res, ModelData.Nz_str,
                                ModelData.Np_str, dataType="Tracked_Profiles", dataName="Tracked_Profiles",
                                dtype='float32',codeSection = "Project_Algorithms")


# In[108]:


#IMPORT CLASSES
sys.path.append(os.path.join(mainCodeDirectory,"3_Project_Algorithms","2_Tracking_Algorithms"))
from CLASSES_TrackingAlgorithms import TrackingAlgorithms_DataLoading_Class, Results_InputOutput_Class, TrackedParcel_Loading_Class


# In[109]:


# IMPORT CLASSES
sys.path.append(os.path.join(mainCodeDirectory,"3_Project_Algorithms","3_Tracked_Profiles"))
from CLASSES_TrackedProfiles import TrackedProfiles_DataLoading_CLASS


# In[110]:


import sys
path=os.path.join(mainCodeDirectory,'Functions/')
sys.path.append(path)

import NumericalFunctions
from NumericalFunctions import * # import NumericalFunctions 
import PlottingFunctions
from PlottingFunctions import * # import PlottingFunctions

# # Get all functions in NumericalFunctions
# import inspect
# functions = [f[0] for f in inspect.getmembers(NumericalFunctions, inspect.isfunction)]
# functions

#####

#Import StatisticalFunctions 
import sys
dir2='/mnt/lustre/koa/koastore/torri_group/air_directory/Projects/DCI-Project/'
path=dir2+'Functions/'
sys.path.append(path)

import StatisticalFunctions
from StatisticalFunctions import * # import NumericalFunctions 


# In[111]:


##############################################
#JOB ARRAY


# In[112]:


#JOB ARRAY SETUP
UsingJobArray=True

def GetNumJobs(res,t_res):
    if res=='1km':
        if t_res=='5min':
            num_jobs=132
        elif t_res=='1min':
            num_jobs=660
    elif res=='250m': 
        if t_res=='1min':
            num_jobs=660
    return num_jobs
num_jobs = GetNumJobs(ModelData.res,ModelData.t_res)
SlurmJobArray = SlurmJobArray_Class(total_elements=ModelData.Ntime, num_jobs=num_jobs, UsingJobArray=UsingJobArray)
start_job = SlurmJobArray.start_job; end_job = SlurmJobArray.end_job

def GetNumElements():
    loop_elements = np.arange(ModelData.Ntime)[start_job:end_job]
    return loop_elements
loop_elements = GetNumElements()


# In[113]:


##############################################
#DATA LOADING FUNCTIONS


# In[114]:


def MakeDataDictionary(variableNames,t,printstatement=False):
    timeString = ModelData.timeStrings[t]
    # print(f"Getting data from {timeString}","\n")
    
    dataDictionary = {variableName: CallLagrangianArray(ModelData, DataManager, timeString, variableName=variableName, printstatement=printstatement) 
                      for variableName in variableNames}      
    return dataDictionary
    
def GetSpatialData(t):    
    variableNames = ['Z','Y','X']
    dataDictionary = MakeDataDictionary(variableNames,t)
    [Z,Y,X] = (dataDictionary[k] for k in variableNames)
    return Z,Y,X

def GetLangrangianBinaryArray(t):
    variableNames=['PROCESSED_A_g','PROCESSED_A_c']
    binaryDictionary = MakeDataDictionary(variableNames,t)
    
    A_g = binaryDictionary['PROCESSED_A_g']
    A_c = binaryDictionary['PROCESSED_A_c']

    return A_g,A_c


# In[115]:


########################################
#RUNNING FUNCTIONS


# In[116]:


#Functions for Initializing Profile Arrays
def CopyStructure(dictionary, placeholder=None):
    """Deep-copy dictionary structure, replacing leaves with a given placeholder."""
    if isinstance(dictionary, dict):
        return {k: CopyStructure(v, placeholder) for k, v in dictionary.items()}
    else:
        return placeholder
        
def InitializeHistograms(trackedArrays, varNames, z_bins,tBins_mins_g,tBins_mins_c, property_bins_Dictionary): #*ZSubsetting
    """
    Create a nested structure matching trackedArrays,
    with empty histogram arrays for each variable:
        - var_hist2d
        - var_parcel_last_time_hist2d
    """
    
    histogramsDictionary = {}
    n_z = len(z_bins) - 1 #*ZSubsetting
    
    for category, depth_dict in trackedArrays.items():  # e.g. 'CL', 'SBF'
        histogramsDictionary[category] = {}

        for depth_type in depth_dict.keys():  # e.g. 'ALL', 'SHALLOW', 'DEEP'
            histogramsDictionary[category][depth_type] = {}

            for varName in varNames:

                # ---- initialize varName level
                histogramsDictionary[category][depth_type][varName] = {}
                for mode in ["g","c"]:
                    histogramsDictionary[category][depth_type][varName][mode] = {}

                # number of property bins for this variable
                n_prop = len(property_bins_Dictionary[varName]) - 1

                # initialized z-subsetted empty histograms
                for mode, tBins in { #*ZSubsetting
                    "g": tBins_mins_g, #*ZSubsetting
                    "c": tBins_mins_c, #*ZSubsetting
                }.items():
                    for t1, t2 in tBins: #*ZSubsetting
                        tKey = f"{t1}_{t2}mins_hist2d" #*ZSubsetting
                        histogramsDictionary[category][depth_type][varName][mode][tKey] = np.zeros( #*ZSubsetting
                            (n_prop, n_prop) #*ZSubsetting #*HeightAverage
                        )

    return histogramsDictionary

def UpdateDictKeys(ZhistogramsDictionary): #*HeightAverage
    d=ZhistogramsDictionary
    for key1 in d:
        for key2 in d[key1]:
            d[key1][key2] = {
                f"QV--{k}": v for k, v in d[key1][key2].items()
            }
    return ZhistogramsDictionary


# In[117]:


def GetParcelNumbers(trackedArray, t):
    """
    Return all parcel indices (p) and their corresponding row indices
    for parcels that are active at time t.
    Vectorized, no row-by-row loops.
    """
    t_start = trackedArray[:, 1]
    t_end   = np.minimum(trackedArray[:, 2] + trackedArray[:, 3], ModelData.Ntime)

    # Boolean mask for rows active at time t
    mask = (t >= t_start) & (t <= t_end)

    # Extract parcel numbers and their corresponding row indices
    selectedRows = np.where(mask)[0]
    selectedPs = trackedArray[selectedRows, 0]
    leftRightIndexes = trackedArray[selectedRows, 4]

    return selectedRows, selectedPs, leftRightIndexes


# In[118]:


#FUNCTIONS FOR GETTING GRID BOX MATCHES

def GetGridBoxMatches_V1(Z,Y,X, zLevels,yLevels,xLevels):
    gridboxMatches = [
        np.where((Z == zLevel) & (Y == yLevel) & (X == xLevel))[0]
        for zLevel, yLevel, xLevel in zip(zLevels, yLevels, xLevels)
    ]
    if len(gridboxMatches) == 0:
        return None
    return gridboxMatches

from collections import defaultdict
def BuildGridboxIndex(Z, Y, X):
    gridIndex = defaultdict(list)
    for i in range(len(Z)):
        gridIndex[(Z[i], Y[i], X[i])].append(i)
    return gridIndex
def GetGridBoxMatches_V2(Z,Y,X, zLevels,yLevels,xLevels):
    gridIndex = BuildGridboxIndex(Z, Y, X)
    gridboxMatches = [
        np.asarray(gridIndex[(z, y, x)], dtype=int)
        for z, y, x in zip(zLevels, yLevels, xLevels)
    ]
    if len(gridboxMatches) == 0:
        return None
    return gridboxMatches


# def CheckIfSame_GridBoxMatches(one,two):
#     same = (
#         len(one) == len(two)
#         and all(
#             np.array_equal(a, b)
#             for a, b in zip(one, two)
#         )
#     )
#     print(same,"#"*10,"\n")

# gridboxMatches_original = GetGridBoxMatches_V1(Z,Y,X, zLevels,yLevels,xLevels)
# gridboxMatches = GetGridBoxMatches_V2(Z,Y,X, zLevels,yLevels,xLevels)
# CheckIfSame_GridBoxMatches(gridboxMatches_original,gridboxMatches)   


# In[119]:


#FUNCTIONS FOR APPLYING CLOUD MASK TO PARCELS

def GetEntrainmentMask(A_g,A_g_Prior,
                       A_c,A_c_Prior,
                       selectedPs):
    mask_g = (A_g & (~A_g_Prior)).astype(bool)
    mask_g[selectedPs] = False #remove the selected parcels themselves
    mask_c = (A_c & (~A_c_Prior)).astype(bool)            
    mask_c[selectedPs] = False #remove the selected parcels themselves
    return mask_g,mask_c
    
# def GetWhereOtherEntrainedParcels_V1(mask_c,gridboxMatches):
#     whereOtherEntrainedParcels_c = [idx[mask_c[idx]]
#                                     for idx in gridboxMatches]
#     if len(whereOtherEntrainedParcels_c) == 0:
#         return None
#     collapsed = np.concatenate(whereOtherEntrainedParcels_c)
#     return collapsed

def GetWhereOtherEntrainedParcels_V2(mask_c,gridboxMatches):
    collapsed = np.concatenate(gridboxMatches)
    collapsed = collapsed[mask_c[collapsed]]
    if collapsed.size == 0:
        return None
    return collapsed
    
# def CheckIfSame_WhereOtherEntrainedParcel(one,two):
#     same = np.array_equal(np.sort(one),
#                           np.sort(two))
#     print(same,"#"*10,"\n")
# collapsed_original = GetWhereOtherEntrainedParcels_V1(mask_c,gridboxMatches)
# collapsed = GetWhereOtherEntrainedParcels_V2(mask_c,gridboxMatches)
# CheckIfSame_WhereOtherEntrainedParcel(collapsed_original,collapsed)


# In[120]:


#FUNCTIONS FOR MAKING PROPERTY HISTOGRAM
def AccumulatePropertyHistogram(histogramsDictionary, #*ZSubsetting
                                ZhistogramsDictionary, #*HeightAverage
                                key1,key2,varName_1,varName_2, #*HeightAverage
                                array_1,array_2,Z, #*HeightAverage
                                collapsed_g,collapsed_c,
                                relative_time,
                                property_bins_Dictionary,
                                z_bins,tBins_mins_g,tBins_mins_c, #*ZSubsetting
                                baselineValues):
    for mode, collapsed, tBins_mins in ( #*ZSubsetting
        ("g", collapsed_g, tBins_mins_g), #*ZSubsetting
        ("c", collapsed_c, tBins_mins_c), #*ZSubsetting
    ):
        if (collapsed is None): continue
            

        #GETTING PROPERTY HISTOGRAMS
        ##########
        # property values at this time for these entrained parcels
        properties_1 = array_1[collapsed] - baselineValues[varName_1][collapsed] #*PERTURBATION #*HeightAverage
        properties_2 = array_2[collapsed] - baselineValues[varName_2][collapsed] #*PERTURBATION #*HeightAverage
        zVals_km = ModelData.zh[Z[collapsed]] #*ZSubsetting
        
        # property bins for each varName
        property_1_bins = property_bins_Dictionary[varName_1] #*HeightAverage
        property_2_bins = property_bins_Dictionary[varName_2] #*HeightAverage
    
        # ==========================================================
        # Make histograms
        # ==========================================================
        for t1,t2 in tBins_mins: #*ZSubsetting
            if CheckRelativeTimeMatch([(t1,t2)],relative_time) is False: #*ZSubsetting
                continue

            property_hist2d_T, _, _ = np.histogram2d( #*ZSubsetting
                properties_1, #*ZSubsetting
                properties_2, #*ZSubsetting
                bins=(property_1_bins, property_2_bins),
                
            )
            Z_hist2d_T, _, _ = np.histogram2d( #*ZSubsetting
                properties_1, #*ZSubsetting
                properties_2, #*ZSubsetting
                bins=(property_1_bins, property_2_bins),
                weights=zVals_km
            )
    
            tKey = f"{t1}_{t2}mins_hist2d" #*ZSubsetting
            histogramsDictionary[key1][key2][f"{varName_1}--{varName_2}"][mode][tKey] += property_hist2d_T #*ZSubsetting
            ZhistogramsDictionary[key1][key2][f"{varName_1}--{varName_2}"][mode][tKey] += Z_hist2d_T #*ZSubsetting

def CheckRelativeTimeMatch(tBins_mins,relative_time): #*ZSubsetting
    if not [(t1, t2) for t1, t2 in tBins_mins 
        if relative_time in np.arange(-int(timesteps_per_min*t1), -int(timesteps_per_min*t2)-1, -1)]: 
        return False
    else:
        return True

# def AccumulatePropertyHistogram(histogramsDictionary, #*ZSubsetting
#                                 key1,key2,varName,
#                                 array,Z,
#                                 collapsed_g,collapsed_c,
#                                 relative_time,
#                                 property_bins_Dictionary,
#                                 z_bins,tBins_mins_g,tBins_mins_c, #*ZSubsetting
#                                 baselineValues):
#     for mode, collapsed, tBins_mins in ( #*ZSubsetting
#         ("g", collapsed_g, tBins_mins_g), #*ZSubsetting
#         ("c", collapsed_c, tBins_mins_c), #*ZSubsetting
#     ):
#         if (collapsed is None): continue
            

#         #GETTING PROPERTY HISTOGRAMS
#         ##########
#         # property values at this time for these entrained parcels
#         properties = array[collapsed] - baselineValues[varName][collapsed] #*PERTURBATION
#         zVals_km = Z[collapsed] #*ZSubsetting
        
#         # property bins for each varName
#         property_bins = property_bins_Dictionary[varName]
    
#         # ==========================================================
#         # Make histograms
#         # ==========================================================
#         for t1,t2 in tBins_mins: #*ZSubsetting
#             if CheckRelativeTimeMatch([(t1,t2)],relative_time) is False: #*ZSubsetting
#                 continue
            
#             property_hist2d_T, _, _ = np.histogram2d( #*ZSubsetting
#                 zVals_km, #*ZSubsetting
#                 properties, #*ZSubsetting
#                 bins=(z_bins, property_bins)
#             )
    
#             tKey = f"{t1}_{t2}mins_hist2d" #*ZSubsetting
#             histogramsDictionary[key1][key2][varName][mode][tKey] += property_hist2d_T #*ZSubsetting


# In[121]:


def MakeTrackedProfiles(trackedArrays,histogramsDictionary,
                        ZhistogramsDictionary, #*HeightAverage
                        property_bins_Dictionary,varNames,
                        Z,Y,X,t, A_g,A_c,A_g_Prior,A_c_Prior,
                        tBins_mins_g,tBins_mins_c, #*ZSubsetting
                        printstatement=True):
    """
    Update profileArraysDictionary with variable data for parcels active at time t.
    Accumulates sums and counts in both profile_array and profile_array_squares.
    """
    
    baselineValues = MakeDataDictionary(varNames, t) #*PERTURBATION
    #CALCULATING
    for key1, subdict in trackedArrays.items():         # e.g. 'CL', 'SBF'
        print("\t",f'working on {key1}')
        for key2, trackedArray in subdict.items():           # e.g. 'ALL', 'DEEP'
            print("\t\t",f'working on {key2}')
    
            #Part 1: getting parcels in trackedArray to run through
            if printstatement: print(f"Part 1: getting parcels in trackedArray to run through")
                
            _, selectedPs, leftRightIndexes = GetParcelNumbers(trackedArray, t) #get parcels that are counted at time t
            if printstatement: print(f"\tRunning for {len(selectedPs)} Parcels")
            
            #getting Z,Y,X data
            zLevels = Z[selectedPs]; yLevels = Y[selectedPs]; xLevels = X[selectedPs]

            #Part 2: find which other parcels exist in each grid box
            if printstatement: print(f"Part 2: find which other parcels exist in each grid box")
                
            # Step a: compute spatial matches once
            if printstatement: print("\tStep a: compute spatial matches once") #SLOW POINT HERE
            gridboxMatches = GetGridBoxMatches_V2(Z,Y,X, zLevels,yLevels,xLevels)
            if gridboxMatches is None:
                continue

            #Part 3: find which of those parcels were entrained into a general/cloudy updraft
            if printstatement: print(f"Part 3: find which of those parcels were entrained into a general/cloudy updraft")
            
            # Step a: compute entrainment masks
            if printstatement: print("\tStep a: compute entrainment masks")
            mask_g,mask_c = GetEntrainmentMask(A_g,A_g_Prior,
                                               A_c,A_c_Prior,
                                               selectedPs)

            # Step b: apply masks to find all parcels
            if printstatement: print("\tStep b: apply masks to find all parcels") #SLOW POINT HERE
            collapsed_g = GetWhereOtherEntrainedParcels_V2(mask_g,gridboxMatches)
            collapsed_c = GetWhereOtherEntrainedParcels_V2(mask_c,gridboxMatches)
            if (collapsed_g is None) and (collapsed_c is None): continue

            # Step c: track parcels back (last 30 minutes) and read properties
            if printstatement: print("\tStep c: track parcels back (last 60 minutes) and read properties")
            
            trackTimes = np.arange(t,(t-timesteps_per_hour)-1,-1)
            for count, t_back in enumerate(tqdm(trackTimes,desc="\t\tTracking back parcels",leave=False)):
                relative_time = t_back - t
                if CheckRelativeTimeMatch(tBins_mins_g,relative_time) is False and CheckRelativeTimeMatch(tBins_mins_c,relative_time) is False: #*ZSubsetting
                    continue

                VARs = MakeDataDictionary(varNames, t_back)   
                for varName_1, varName_2 in varPairs: #*HeightAverage
                    array_1 = VARs[varName_1]; array_2 = VARs[varName_2] #*HeightAverage
                    #GETTING PROPERTY HISTOGRAMS
                    AccumulatePropertyHistogram(histogramsDictionary,
                                                ZhistogramsDictionary, #*HeightAverage
                                                key1,key2,varName_1,varName_2, #*HeightAverage
                                                array_1,array_2,Z, #*HeightAverage
                                                collapsed_g,collapsed_c,
                                                relative_time,
                                                property_bins_Dictionary,
                                                z_bins,tBins_mins_g,tBins_mins_c, #*ZSubsetting
                                                baselineValues) #*PERTURBATION

    return histogramsDictionary,ZhistogramsDictionary


# In[122]:


########################################
#RUNNING


# In[123]:


def MakeSignedLogBins(minAbs, maxAbs, nBins):
    """
    Create symmetric signed-logarithmic bins including zero.
    """

    if nBins % 2 != 0:
        raise ValueError("nBins must be even for signed-log bins")

    halfBins = nBins // 2

    # positive side (log-spaced, excludes zero)
    pos = np.logspace(
        np.log10(minAbs),
        np.log10(maxAbs),
        halfBins
    )

    # negative side (mirror)
    neg = -pos[::-1]

    # include zero explicitly
    bins = np.concatenate([neg, [0.0], pos])

    return bins


# In[124]:


#Loading in Tracked Parcels Info
trackedArrays,LevelsDictionary = TrackedParcel_Loading_Class.LoadingSubsetParcelData(ModelData,DataManager,
                                                         Results_InputOutput_Class)
trackedArrays.pop("ColdPool") #removing this extra unneeded category

#needed parameters
timesteps_per_min = 1/(ModelData.time[1].item()/1e9/60 )
timesteps_per_hour = int(60*timesteps_per_min)
qcqi_thresh = 1e-6
# time_bins = np.arange(0,(0-timesteps_per_hour)-1,-1)[::-1]
time_bins = np.arange(0.5, -timesteps_per_hour-1.5, -1)[::-1]
z_bins = np.arange(0.5, len(ModelData.zh)+1.5, 1) #*ZSubsetting

#variables 
varNames = ["QV", "QCQI", "W", "THETA_v"]

#property bins for each variable
n_bins = 500
# property_bins_Dictionary = {
#     "QV":    np.linspace(-5/1e3, 5/1e3, n_bins),        # water vapor mixing ratio
#     "QCQI":  np.linspace(-3e-3, 3e-3, n_bins),         # cloud+ice mixing ratio
#     "W":     np.linspace(-15, 15, n_bins),         # vertical velocity bins
#     "THETA_v":    np.linspace(-10, 10, n_bins),       # potential temperature
# }
property_bins_Dictionary = {
    "QV":    MakeSignedLogBins(5e-5, 5/1e3, n_bins),        # water vapor mixing ratio perturbation
    "QCQI":  MakeSignedLogBins(1e-8, 3/1e3, n_bins),         # cloud+ice mixing ratio perturbation
    "W":     MakeSignedLogBins(0.02, 15, n_bins),         # vertical velocity bins perturbation
    "THETA_v":    MakeSignedLogBins(0.1, 10, n_bins),       # potential temperature perturbation
}

#trackback times to bin for (must be less than 60 minutes)
tBins_mins_g = [(10, 10), #*ZSubsetting
                (20, 20),
                (30, 30)]
tBins_mins_c = [(10, 10), #*ZSubsetting
                (20, 20),
                (30, 30)]

varPairs = [ #*HeightAverage
    ("QV", "W"),
    ("QV", "QCQI"),
    ("QV", "THETA_v"),
]


# In[ ]:


for t in tqdm(loop_elements, desc="Processing"):
    if t <= timesteps_per_hour:
        print(f"skipping time {t} since too close to first hour")
        continue
        
    print("#" * 40,"\n",f"Processing timestep {t}/{loop_elements[-1]}")
    timeString = ModelData.timeStrings[t]

    #Forming Dictionary for Profile Arrays for current timestep
    trackedProfileArrays = CopyStructure(trackedArrays)
    histogramsDictionary = InitializeHistograms(trackedProfileArrays,varNames, z_bins,tBins_mins_g,tBins_mins_c, property_bins_Dictionary); histogramsDictionary = UpdateDictKeys(histogramsDictionary) #*HeightAverage
    ZhistogramsDictionary = InitializeHistograms(trackedProfileArrays,varNames, z_bins,tBins_mins_g,tBins_mins_c, property_bins_Dictionary); ZhistogramsDictionary = UpdateDictKeys(ZhistogramsDictionary) #*HeightAverage
    
    #getting variable data
    Z,Y,X = GetSpatialData(t)
    A_g,A_c = GetLangrangianBinaryArray(t)
    A_g_Prior,A_c_Prior = GetLangrangianBinaryArray(t-1)
    
    #making tracked profiles
    print("MAKING TRACKED PROFILES")
    [histogramsDictionary,ZhistogramsDictionary] = MakeTrackedProfiles(trackedArrays,histogramsDictionary,
                                               ZhistogramsDictionary, #*HeightAverage
                                               property_bins_Dictionary,varNames,
                                               Z,Y,X,t, A_g,A_c,A_g_Prior,A_c_Prior,
                                               tBins_mins_g,tBins_mins_c) #*ZSubsetting
    
    #saving tracked profiles for current timestep
    TrackedProfiles_DataLoading_CLASS.SaveProfile(ModelData,DataManager_TrackedProfiles, histogramsDictionary, dataName="EntrainmentTrackback_Perturbation_ZSubsetting_HeightAverage", t=t) #*ZSubsetting #*HeightAverage
    TrackedProfiles_DataLoading_CLASS.SaveProfile(ModelData,DataManager_TrackedProfiles, ZhistogramsDictionary, dataName="EntrainmentTrackback_Perturbation_ZSubsetting_HeightAverage_Z", t=t) #*ZSubsetting #*HeightAverage


# In[310]:


#########################################
#RECOMBINE SEPERATE JOB_ARRAYS AFTER
recombine=False #KEEP FALSE WHEN JOBARRAY IS RUNNING
recombine=True


# In[311]:


import copy
def RecombineProfiles(ModelData, DataManager):
    """
    Combine tracked profiles across all timesteps using the first as a template.
    """
    print(f"Recombining {ModelData.Ntime} TrackedProfiles files...\n")

    histogramsDictionary_combined = None

    for t in tqdm(range(ModelData.Ntime), desc="Combining Profiles", unit="timestep"):

        if t <= timesteps_per_hour:
            print(f"skipping time {t} since too close to first hour")
            continue
        
        histogramsDictionary = TrackedProfiles_DataLoading_CLASS.LoadProfile(ModelData, DataManager, dataName="EntrainmentTrackback_Perturbation_ZSubsetting_HeightAverage", t=t)
        ZhistogramsDictionary = TrackedProfiles_DataLoading_CLASS.LoadProfile(ModelData, DataManager, dataName="EntrainmentTrackback_Perturbation_ZSubsetting_HeightAverage_Z", t=t)
         
        # --- initialize on first timestep ---
        if histogramsDictionary_combined is None:
            histogramsDictionary_combined = copy.deepcopy(histogramsDictionary)
            ZhistogramsDictionary_combined = copy.deepcopy(ZhistogramsDictionary)
            continue
    
        # --- accumulate later timesteps ---
        for key1 in histogramsDictionary:
            for key2 in histogramsDictionary[key1]:
                for varName in histogramsDictionary[key1][key2]:
                    for mode in histogramsDictionary[key1][key2][varName]:
                        for tKey in histogramsDictionary[key1][key2][varName][mode]:
                            
                            # counts
                            histogramsDictionary_combined[key1][key2][varName][mode][tKey] += (
                                histogramsDictionary[key1][key2][varName][mode][tKey]
                            )
        
                            # z-sum
                            ZhistogramsDictionary_combined[key1][key2][varName][mode][tKey] += (
                                ZhistogramsDictionary[key1][key2][varName][mode][tKey]
                            )
    return histogramsDictionary_combined,ZhistogramsDictionary_combined


# In[ ]:


if recombine==True:
    [histogramsDictionary_combined,ZhistogramsDictionary_combined] = RecombineProfiles(ModelData, DataManager_TrackedProfiles)
    TrackedProfiles_DataLoading_CLASS.SaveProfile(ModelData,DataManager_TrackedProfiles, 
                                                  histogramsDictionary_combined, dataName="EntrainmentTrackback_Perturbation_ZSubsetting_HeightAverage", t='combined') #*ZSubsetting #*HeightAverage
    TrackedProfiles_DataLoading_CLASS.SaveProfile(ModelData,DataManager_TrackedProfiles, 
                                                  ZhistogramsDictionary_combined, dataName="EntrainmentTrackback_Perturbation_ZSubsetting_HeightAverage_Z", t='combined') #*ZSubsetting #*HeightAverage


# In[ ]:










# In[313]:


###################
#PLOTTING FUNCTIONS
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

plotting=False
plotting=True


# In[321]:


#Loading Back In
if plotting:
    histogramsDictionary_combined = TrackedProfiles_DataLoading_CLASS.LoadProfile(ModelData,DataManager_TrackedProfiles, dataName="EntrainmentTrackback_Perturbation_ZSubsetting_HeightAverage", t='combined') #*ZSubsetting #*HeightAverage
    ZhistogramsDictionary_combined = TrackedProfiles_DataLoading_CLASS.LoadProfile(ModelData,DataManager_TrackedProfiles, dataName="EntrainmentTrackback_Perturbation_ZSubsetting_HeightAverage_Z", t='combined') #*ZSubsetting #*HeightAverage


# In[299]:


def CombinedPlot_PropertyHistogram_V1(parcelType="CL",mode="c",tKey="20_20mins_hist2d"):
    
    varNames = ["QCQI", "W", "THETA_v"]
    parcelDepths = ["ALL", "SHALLOW", "DEEP"]
    
    unit_map = {"QV": "g/kg","QCQI": "g/kg","W": "m/s","THETA_v": "K"}
    
    # --- figure + gridspec ---
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05, hspace=0.15)
    
    # --- colormap ---
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_under("white")
    
    # --- shared levels ---
    if mode=="c":
        levels = np.arange(0, 6.01, 0.25)
    elif mode=="g":
        levels = np.arange(0, 1.01, 0.05)
    
    axes = []
    
    for i, parcelDepth in enumerate(parcelDepths):
        row_axes = []
        for j, varName in enumerate(varNames):
    
            # --- shared y per row ---
            if j == 0:
                ax = fig.add_subplot(gs[i, j])
            else:
                ax = fig.add_subplot(gs[i, j], sharey=row_axes[0])
    
            row_axes.append(ax)
    
            multiplier = 1e3 if "Q" in varName else 1
            units = unit_map.get(varName, "")
    
            countHist = histogramsDictionary_combined[parcelType][parcelDepth][f'QV--{varName}'][mode][tKey]
            zSumHist  = ZhistogramsDictionary_combined[parcelType][parcelDepth][f'QV--{varName}'][mode][tKey]
    
            # --- safe divide ---
            zMean = np.divide(
                zSumHist,
                countHist,
                out=np.zeros_like(zSumHist, dtype=float),
                where=countHist != 0
            )
    
            # --- bins ---
            x_bins = property_bins_Dictionary[varName] * multiplier
            y_bins = property_bins_Dictionary["QV"] * 1e3
    
            x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
            y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
    
            # --- plot ---
            im = ax.contourf(
                x_centers,
                y_centers,
                zMean,
                cmap=cmap,
                levels=levels,
                extend='both'
            )
    
            # --- labels ---
            if i == 2:  # only bottom row
                ax.set_xlabel(fr"${varName}'\ ({units})$")
    
            if j == 0:  # only left column
                ax.set_ylabel(parcelDepth+f"\n"+r"$QV'\ (g/kg)$")
    
            ax.axvline(0, color='black', linestyle='--', alpha=0.6, linewidth=1)
            ax.axhline(0, color='black', linestyle='--', alpha=0.6, linewidth=1)
    
            # --- titles ---
            if i == 0:
                ax.set_title(f"{varName}")
    
        # --- hide duplicate y ticks ---
        for ax in row_axes[1:]:
            ax.tick_params(labelleft=False)
    
        axes.append(row_axes)
    
    # --- colorbar ---
    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\overline{Z}\ (km)$")
    
    # --- title ---
    plt.suptitle(
        f"Parcel History Histograms ({parcelType}) ({tKey.replace('_', '-', 1).split('_')[0]})",
        fontsize=16,
        y=0.93
    )

    return fig


# In[304]:


def GetPlottingDirectory(plotFileName, plotType):
    plottingDirectory = mainCodeDirectory=os.path.join(mainDirectory,"Code","PLOTTING")
    
    specificPlottingDirectory = os.path.join(plottingDirectory, plotType, 
                                             f"{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz")
    os.makedirs(specificPlottingDirectory, exist_ok=True)

    plottingFileName=os.path.join(specificPlottingDirectory, plotFileName)

    return plottingFileName
    
def SaveFigure(fig,fileName,
               plotType=f"Project_Algorithms/Tracked_Profiles/Tracked_Profiles_EntrainmentTrackback_Perturbation_ZSubsetting_HeightAverage"): #*HeightAverage
    plotFileName = f"{fileName}_{ModelData.res}_{ModelData.t_res}_{ModelData.Np_str}.jpg"
    plottingFileName = GetPlottingDirectory(plotFileName, plotType)
    
    print(f"Saving figure to {plottingFileName}")
    fig.savefig(plottingFileName, dpi=300, bbox_inches='tight')
    plt.close(fig) 


# In[305]:


###################
#PLOTTING


# In[ ]:


if plotting:
    parcelTypes = ["CL","nonCL","SBF"]
    for parcelType in tqdm(parcelTypes):
        for mode, tBins_mins in ( #*ZSubsetting
                ("g", tBins_mins_g), #*ZSubsetting
                ("c", tBins_mins_c)): #*ZSubsetting
            
            for t1, t2 in tBins_mins: #*ZSubsetting
                tKey = f"{t1}_{t2}mins_hist2d" #*ZSubsetting
                fig = CombinedPlot_PropertyHistogram_V1(parcelType,mode,tKey) #*ZSubsetting
                fileName = f"EntrainmentTrackback_Perturbation_ZSubsetting_HeightAverage_PropertyHistogram_{parcelType}_{mode}_{tKey}" #*ZSubsetting #*HeightAverage
                SaveFigure(fig,fileName)


# In[308]:





# In[ ]:





# In[ ]:





# In[ ]:


###################
#TESTING


# In[ ]:


# def PlotSpecific(mode="c", #g
#                  parcelType="CL", #nonCL, SBF
#                  t1=30): #10,20
#     t2=t1
#     tKey = f"{t1}_{t2}mins_hist2d" #*ZSubsetting
#     fig = CombinedPlot_PropertyHistogram_V1(parcelType,mode,tKey) #*ZSubsetting

# PlotSpecific(parcelType="CL",t1=10)
# PlotSpecific(parcelType="CL",t1=20)
# PlotSpecific(parcelType="CL",t1=30)
# # PlotSpecific(parcelType="nonCL",t1=30)

