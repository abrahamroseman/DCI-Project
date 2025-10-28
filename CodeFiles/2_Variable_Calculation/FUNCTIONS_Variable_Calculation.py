#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ============================================================
# CallVariable_Function
# ============================================================

import os

def CallVariable(ModelData, DataManager, timeString, variableName, zInterpolate = None):
    if variableName in ModelData.varList:
        var_data = DataManager.GetTimestepData(DataManager.inputDataDirectory, timeString, 
                                               variableName=variableName, zInterpolate = zInterpolate)
        
    elif variableName not in ModelData.varList:
        if variableName in ["A_g","A_c","qcqi"]:
            dataType = "CalculateMoreVariables"
            dataName = "Eulerian_Binary_Array"
            dataFolder = dataName
        elif variableName in ["VMF_g","VMF_c"]:
            dataType = "CalculateMoreVariables"
            dataName = "Eulerian_VMF"
            dataFolder = dataName
        elif variableName in ["MSE"]:
            dataType = "CalculateMoreVariables"
            dataName = "Moist_Static_Energy"
            dataFolder = dataName
        elif variableName in ["theta_v"]:
            dataType = "CalculateMoreVariables"
            dataName = "Virtual_Potential_Temperature"
            dataFolder = dataName
        elif variableName in ["theta_e", "RH_vapor", "RH_ice"]:
            dataType = "CalculateMoreVariables"
            dataName = "Equivalent_Potential_Temperature"
            dataFolder = dataName
        elif variableName in ["convergence"]:
            dataType = "CalculateMoreVariables"
            dataName = "Convergence"
            dataFolder = dataName
        elif variableName in ["HMC"]:
            dataType = "CalculateMoreVariables"
            dataName = "MoistureConvergence"
            dataFolder = dataName

        elif variableName in ['Entrainment_g','Entrainment_c',
                              'TransferEntrainment_g',
                              'TransferEntrainment_c']:
            dataType = "EntrainmentCalculation"
            dataName = "Entrainment"
            dataFolder = "EntrainmentCalculation"

        elif variableName in ['Detrainment_g','Detrainment_c',
                              'TransferDetrainment_g',
                              'TransferDetrainment_c']:
            dataType = "EntrainmentCalculation"
            dataName = "Detrainment"
            dataFolder = "EntrainmentCalculation"

        elif variableName in ['PROCESSED_Entrainment_g','PROCESSED_Entrainment_c',
                              'PROCESSED_TransferEntrainment_g',
                              'PROCESSED_TransferEntrainment_c']:
            dataType = "EntrainmentCalculation"
            dataName = "PROCESSED_Entrainment"
            dataFolder = "EntrainmentCalculation"
    
        elif variableName in ['PROCESSED_Detrainment_g','PROCESSED_Detrainment_c',
                              'PROCESSED_TransferDetrainment_g',
                              'PROCESSED_TransferDetrainment_c']:
            dataType = "EntrainmentCalculation"
            dataName = "PROCESSED_Detrainment"
            dataFolder = "EntrainmentCalculation"

            
        inputDataDirectory = os.path.normpath(
            os.path.join(DataManager.inputDirectory, "..", dataType,
                         f"{DataManager.res}_{DataManager.t_res}_{DataManager.Nz_str}nz", dataFolder)
                        )
        var_data = DataManager.GetTimestepData(inputDataDirectory, timeString, 
                                               variableName=variableName, dataName=dataName)
    return var_data


# In[6]:


# ============================================================
# CallLagrangianArray_Function
# ============================================================

import os

def CallLagrangianArray(ModelData, DataManager, timeString, variableName, 
                        printstatement=False):

    if variableName in ["A_g","A_c","z","x","Z","Y","X","qcqi"]:
        dataType = "LagrangianArrays"
        dataName = "Lagrangian_Binary_Array"
    elif variableName in ["PROCESSED_A_g","PROCESSED_A_c"]:
        dataType = "LagrangianArrays"
        dataName = "PROCESSED_Lagrangian_Binary_Array"
    elif variableName in ["VMF_g","VMF_c"]:
        dataType = "LagrangianArrays"
        dataName = "Eulerian_VMF"
    elif variableName in ["MSE"]:
        dataType = "LagrangianArrays"
        dataName = "Moist_Static_Energy"
    elif variableName in ["theta_v"]:
        dataType = "LagrangianArrays"
        dataName = "Virtual_Potential_Temperature"
    elif variableName in ["theta_e", "RH_vapor", "RH_ice"]:
        dataType = "LagrangianArrays"
        dataName = "Equivalent_Potential_Temperature"
    elif variableName in ["convergence"]:
        dataType = "LagrangianArrays"
        dataName = "Convergence"
    elif variableName in ["HMC"]:
        dataType = "LagrangianArrays"
        dataName = "MoistureConvergence"

    elif variableName in ['Entrainment_g','Entrainment_c',
                          'TransferEntrainment_g',
                          'TransferEntrainment_c']:
        dataType = "EntrainmentCalculation"
        dataName = "Entrainment"
        dataFolder = "EntrainmentCalculation"

    elif variableName in ['Detrainment_g','Detrainment_c',
                          'TransferDetrainment_g',
                          'TransferDetrainment_c']:
        dataType = "EntrainmentCalculation"
        dataName = "Detrainment"
        dataFolder = "EntrainmentCalculation"

    elif variableName in ['PROCESSED_Entrainment_g','PROCESSED_Entrainment_c',
                          'PROCESSED_TransferEntrainment_g',
                          'PROCESSED_TransferEntrainment_c']:
        dataType = "EntrainmentCalculation"
        dataName = "PROCESSED_Entrainment"
        dataFolder = "EntrainmentCalculation"

    elif variableName in ['PROCESSED_Detrainment_g','PROCESSED_Detrainment_c',
                          'PROCESSED_TransferDetrainment_g',
                          'PROCESSED_TransferDetrainment_c']:
        dataType = "EntrainmentCalculation"
        dataName = "PROCESSED_Detrainment"
        dataFolder = "EntrainmentCalculation"
        
    inputDataDirectory = os.path.normpath(
        os.path.join(DataManager.inputDirectory, "..", dataType,
                     f"{DataManager.res}_{DataManager.t_res}_{DataManager.Nz_str}nz", dataName))
    var_data = DataManager.GetTimestepData(inputDataDirectory, timeString,
                                           variableName=variableName, dataName=dataName,
                                           printstatement=printstatement)
    return var_data


# In[ ]:


# ============================================================
# Get_LagrangianArrays_Function
# ============================================================

def Get_LagrangianArrays(t, dataType="VARS", dataName="VARS", varNames=["W"]):
    res = ModelData.res
    t_res = ModelData.t_res
    Nz_str = ModelData.Nz_str
    inputDirectory = os.path.join(DataManager.inputDirectory,
                                  "..","LagrangianArrays",
                                  f"{res}_{t_res}_{Nz_str}nz", dataType)
    timeString = ModelData.timeStrings[t]

    FileName = os.path.join(inputDirectory, f"{dataName}_{res}_{t_res}_{Nz_str}nz_{timeString}.h5")

    dataDictionary = {}
    with h5py.File(FileName, 'r') as f:
        # print("Keys in file:", list(f.keys()))
        for key in varNames:
            dataDictionary[key] = f[key][:]
            # print(f"{key}: shape = {dataDictionary[key].shape}, dtype = {dataDictionary[key].dtype}")
    return dataDictionary


# In[ ]:


# ============================================================
# OpenMultipleSingleTimes_LagrangianArray_FUNCTION
# ============================================================

import os
from glob import glob
import xarray as xr

def OpenMultipleSingleTimes_LagrangianArray(directory, ModelData, pattern="Lagrangian_Binary_Array_*.h5"):
    """
    Load a sequence of Lagrangian .h5 files (each a single timestep)
    into one xarray.Dataset with dimensions (time, p),
    enforcing time order from ModelData.timeStrings.
    """
    # --- Find all available files
    files_all = glob(os.path.join(directory, pattern))
    if not files_all:
        raise FileNotFoundError(f"No files found in {directory} matching {pattern}")

    # --- Build the correctly ordered list according to ModelData.timeStrings
    files = []
    for t in ModelData.timeStrings:
        time_pattern = f"_{t}.h5"
        matched = [f for f in files_all if f.endswith(time_pattern)]
        if matched:
            files.append(matched[0])
        else:
            print(f"Missing file for time {t}")

    # --- Open and concatenate along time
    ds = xr.open_mfdataset(
        files,
        engine="h5netcdf",
        phony_dims="sort",
        combine="nested",
        concat_dim="time",
    )

    # --- Rename the phony dimension to 'p'
    if "phony_dim_0" in ds.dims:
        ds = ds.rename({"phony_dim_0": "p"})

    return ds, files

# #EXAMPLE USAGE
# directory = f"/mnt/lustre/koa/koastore/torri_group/air_directory/Projects/DCI-Project/Code/OUTPUT/Variable_Calculation/LagrangianArrays/{ModelData.res}_{ModelData.t_res}_{ModelData.Nz_str}nz/Lagrangian_Binary_Array/"

# Lagrangian_Binary_Array,files = OpenMultipleSingleTimes_LagrangianArray(directory, ModelData)

