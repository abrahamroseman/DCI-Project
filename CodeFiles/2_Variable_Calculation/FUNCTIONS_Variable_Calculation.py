#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ============================================================
# CallVariable_Function
# ============================================================

import os

def CallVariable(ModelData, DataManager, timeString, variableName):
    if variableName in ModelData.varList:
        var_data = DataManager.GetTimestepData(DataManager.inputDataDirectory, timeString, variableName=variableName)
        
    elif variableName not in ModelData.varList:
        if variableName in ["A_g","A_c"]:
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
            dataName = "Eulerian_Binary_Array"
        elif variableName in ["VMF_g","VMF_c"]:
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
            dataName = "Eulerian_VMF"
        elif variableName in ["MSE"]:
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
            dataName = "Moist_Static_Energy"
        elif variableName in ["theta_v"]:
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
            dataName = "Virtual_Potential_Temperature"
        elif variableName in ["theta_e", "RH_vapor", "RH_ice"]:
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
            dataName = "Equivalent_Potential_Temperature"
        elif variableName in ["convergence"]:
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
            dataName = "Convergence"
        elif variableName in ["HMC"]:
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
            dataName = "MoistureConvergence"

        elif variableName in ['Entrainment_g','Entrainment_c',
                              'TransferEntrainment_g',
                              'TransferEntrainment_c']:
            dataType = "EntrainmentCalculation"
            dataFolder = dataType
            dataName = "Entrainment"

        elif variableName in ['Detrainment_g','Detrainment_c',
                              'TransferDetrainment_g',
                              'TransferDetrainment_c']:
            dataType = "EntrainmentCalculation"
            dataFolder = dataType
            dataName = "Detrainment"

        elif variableName in ['PROCESSED_Entrainment_g','PROCESSED_Entrainment_c',
                              'PROCESSED_TransferEntrainment_g',
                              'PROCESSED_TransferEntrainment_c']:
            dataType = "EntrainmentCalculation"
            dataFolder = dataType
            dataName = "PROCESSED_Entrainment"
    
        elif variableName in ['PROCESSED_Detrainment_g','PROCESSED_Detrainment_c',
                              'PROCESSED_TransferDetrainment_g',
                              'PROCESSED_TransferDetrainment_c']:
            dataType = "EntrainmentCalculation"
            dataFolder = dataType
            dataName = "PROCESSED_Detrainment"

            
        inputDataDirectory = os.path.normpath(
            os.path.join(DataManager.outputDirectory, "..", dataType,
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

    if variableName in ["A_g","A_c","z","x","Z","Y","X"]:
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
        dataFolder = dataType
        dataName = "Entrainment"

    elif variableName in ['Detrainment_g','Detrainment_c',
                          'TransferDetrainment_g',
                          'TransferDetrainment_c']:
        dataType = "EntrainmentCalculation"
        dataFolder = dataType
        dataName = "Detrainment"

    elif variableName in ['PROCESSED_Entrainment_g','PROCESSED_Entrainment_c',
                          'PROCESSED_TransferEntrainment_g',
                          'PROCESSED_TransferEntrainment_c']:
        dataType = "EntrainmentCalculation"
        dataFolder = dataType
        dataName = "PROCESSED_Entrainment"

    elif variableName in ['PROCESSED_Detrainment_g','PROCESSED_Detrainment_c',
                          'PROCESSED_TransferDetrainment_g',
                          'PROCESSED_TransferDetrainment_c']:
        dataType = "EntrainmentCalculation"
        dataFolder = dataType
        dataName = "PROCESSED_Detrainment"
        
    inputDataDirectory = os.path.normpath(
        os.path.join(DataManager.outputDirectory, "..", dataType,
                     f"{DataManager.res}_{DataManager.t_res}_{DataManager.Nz_str}nz", dataName))
    var_data = DataManager.GetTimestepData(inputDataDirectory, timeString,
                                           variableName=variableName, dataName=dataName)
    return var_data

