#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# CallVariable_Function
# ============================================================

import os

def CallVariable(ModelData, DataManager, timeString, variableName):
    if variableName in ModelData.varList:
        var_data = DataManager.GetTimestepData(DataManager.inputDataDirectory, timeString, variableName=variableName)
        
    elif variableName not in ModelData.varList:
        if variableName in ["A_g","A_c"]:
            dataName = "Eulerian_Binary_Array"
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
        elif variableName in ["VMF_g","VMF_c"]:
            dataName = "Eulerian_VMF"
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
        elif variableName in ["MSE"]:
            dataName = "Moist_Static_Energy"
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
        elif variableName in ["theta_v"]:
            dataName = "Virtual_Potential_Temperature"
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
        elif variableName in ["theta_e", "RH_vapor", "RH_ice"]:
            dataName = "Equivalent_Potential_Temperature"
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
        elif variableName in ["convergence"]:
            dataName = "Convergence"
            dataType = "CalculateMoreVariables"
            dataFolder = dataName
        elif variableName in ["HMC"]:
            dataName = "MoistureConvergence"
            dataType = "CalculateMoreVariables"
            dataFolder = dataName

        elif variableName in ['Entrainment_g','Entrainment_c',
                              'TransferEntrainment_g',
                              'TransferEntrainment_c']:
            dataName = "Entrainment"
            dataType = "EntrainmentCalculation"
            dataFolder = dataType

        elif variableName in ['Detrainment_g','Detrainment_c',
                              'TransferDetrainment_g',
                              'TransferDetrainment_c']:
            dataName = "Detrainment"
            dataType = "EntrainmentCalculation"
            dataFolder = dataType
            
        inputDataDirectory = os.path.normpath(
            os.path.join(DataManager.outputDirectory, "..", dataType,
                         f"{DataManager.res}_{DataManager.t_res}_{DataManager.Nz_str}nz", dataFolder)
                        )
        var_data = DataManager.GetTimestepData(inputDataDirectory, timeString, 
                                               variableName=variableName, dataName=dataName)
    return var_data


# In[4]:


# ============================================================
# CallLagrangianArray_Function
# ============================================================

import os

def CallLagrangianArray(ModelData, DataManager, timeString, variableName):

    if variableName in ["A_g","A_c","z","x","Z","Y","X"]:
        dataName = "Lagrangian_Binary_Array"
        dataType = "LagrangianArrays"
    elif variableName in ["VMF_g","VMF_c"]:
        dataName = "Eulerian_VMF"
        dataType = "LagrangianArrays"
    elif variableName in ["MSE"]:
        dataName = "Moist_Static_Energy"
        dataType = "LagrangianArrays"
    elif variableName in ["theta_v"]:
        dataName = "Virtual_Potential_Temperature"
        dataType = "LagrangianArrays"
    elif variableName in ["theta_e", "RH_vapor", "RH_ice"]:
        dataName = "Equivalent_Potential_Temperature"
        dataType = "LagrangianArrays"
    elif variableName in ["convergence"]:
        dataName = "Convergence"
        dataType = "LagrangianArrays"
    elif variableName in ["HMC"]:
        dataName = "MoistureConvergence"
        dataType = "LagrangianArrays"

    elif variableName in ['Entrainment_g','Entrainment_c',
                          'TransferEntrainment_g',
                          'TransferEntrainment_c']:
        dataName = "Entrainment"
        dataType = "EntrainmentCalculation"
        dataFolder = dataType

    elif variableName in ['Detrainment_g','Detrainment_c',
                          'TransferDetrainment_g',
                          'TransferDetrainment_c']:
        dataName = "Detrainment"
        dataType = "EntrainmentCalculation"
        dataFolder = dataType
        
    inputDataDirectory = os.path.normpath(
        os.path.join(DataManager.outputDirectory, "..", dataType,
                     f"{DataManager.res}_{DataManager.t_res}_{DataManager.Nz_str}nz", dataName))
    var_data = DataManager.GetTimestepData(inputDataDirectory, timeString,
                                           variableName=variableName, dataName=dataName)
    return var_data


# In[ ]:




