#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

def CallLagrangianArray(ModelData, DataManager, timeString, variableName):

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
                                           variableName=variableName, dataName=dataName)
    return var_data

