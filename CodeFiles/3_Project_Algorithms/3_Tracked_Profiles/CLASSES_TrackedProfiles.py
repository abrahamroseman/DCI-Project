#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ============================================================
# TrackedProfiles_DataLoading_CLASS
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

    @staticmethod
    def ExtractProfileStandardErrorArrays(profileDict,ProfileStandardError):
        """
        From a nested dictionary like trackedProfileArrays, compute standard error arrays
        using ProfileStandardError(profile_array, profile_array_squares) for each variable,
        and return a new dictionary with the same structure, but only 'profile_array_SE'.
        """
        output = {}
    
        for category, depth_dict in profileDict.items():
            output[category] = {}
    
            for depth, var_dict in depth_dict.items():
                output[category][depth] = {}
    
                for varName, arrays in var_dict.items():
                    profile     = arrays.get("profile_array")
                    profile_sq  = arrays.get("profile_array_squares")
    
                    if profile is not None and profile_sq is not None:
                        profile_SE = ProfileStandardError(profile, profile_sq)
                        output[category][depth][varName] = {
                            "profile_array_SE": profile_SE
                        }
        return output


#Example Call
#IMPORT CLASSES
# sys.path.append(os.path.join(mainCodeDirectory,"3_Project_Algorithms","3_Tracked_Profiles"))
# from CLASSES_TrackedProfiles import TrackedProfiles_DataLoading_CLASS

#Example Run
#saving
# TrackedProfiles_DataLoading_CLASS.SaveProfile(ModelData,DataManager_TrackedProfiles, profileArraysDictionary, t)
#loading
# TrackedProfiles_DataLoading_CLASS.LoadProfile(ModelData,DataManager_TrackedProfiles, t)


# In[ ]:





# In[ ]:


# ============================================================
# TrackedProfiles_Plotting_CLASS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

class TrackedProfiles_Plotting_CLASS:

    @staticmethod
    def ProfileMean(profile): 
        """
        Input Requires Three Column Array 
        with Sum in 1st Column, Count in 2nd Column, and Index in 3rd Column
        Returns 1st and 3rd Column (removing zero rows)
        """
        #gets rid of rows that have no data
        profile_mean=profile[ (profile[:, 1] > 1)]; 
        #divides the data column by the counter column
        profile_mean=np.array([profile_mean[:, 0] / profile_mean[:, 1], profile_mean[:, 2]]).T 
        return profile_mean

    # === Category and depth styles ===
    category_styles = {"CL": "solid", "nonCL": "dashed", "SBF": "dashdot"}
    depth_colors = {"SHALLOW": "green", "DEEP": "blue"}

    @staticmethod
    def PlotSE(axis, profile, SE_profile, color, multiplier=1, switch=1, alpha=0.1, min_value=None):
        lower = multiplier * profile[:, 0] - multiplier * SE_profile[:, 0] * switch
        upper = multiplier * profile[:, 0] + multiplier * SE_profile[:, 0] * switch
        
        if min_value is not None:
            lower = np.maximum(lower, min_value)
        axis.fill_betweenx(profile[:, -1], lower, upper, color=color, alpha=alpha)

    @staticmethod
    def GetHLines(LevelsDictionary):
        hLine_1 = LevelsDictionary["min_all_cloudbase"]
        hLine_2 = LevelsDictionary["MeanLFC"]
    
        hLines = (hLine_1,hLine_2)
        hLineColors = ("purple","#FF8C00")
        return hLines,hLineColors

    @staticmethod
    def PlotHLines(axis,hLines,hLineColors):
        for (hLine,hLineColor) in zip(hLines,hLineColors):
            axis.axhline(hLine, color=hLineColor, linestyle='dashed', zorder=-10)

    @staticmethod
    def ApplyXLimFromZLim(axis, zlim, buffer=0.05):
        """
        Adjust the x-limits of the axis by examining all lines plotted on it.
        Only considers x-values where y is within the zlim range.
        """
        x_all = []
        y_all = []
    
        for line in axis.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            x_all.append(xdata)
            y_all.append(ydata)
    
        if not x_all or not y_all:
            return  # No lines to process
    
        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)
    
        mask = (y_all >= zlim[0]) & (y_all <= zlim[1])
        if np.any(mask):
            xmin = np.min(x_all[mask])
            xmax = np.max(x_all[mask])
            delta = xmax - xmin if xmax > xmin else xmax * buffer
            axis.set_xlim(xmin - delta * buffer, xmax + delta * buffer)
    
        axis.set_ylim(zlim)

    @staticmethod
    def AddCategoryLegend(fig, parcelTypes=["CL", "nonCL", "SBF"], loc='upper center', bbox=(0.5, 0.93)):
        """
        Adds a custom legend for parcel types based on linestyle (e.g., CL, nonCL, SBF).
        """
        linestyle_map = {
            "CL": "solid",
            "nonCL": "dashed",
            "SBF": "dashdot"
        }
    
        custom_lines = [
            Line2D([0], [0], color='black', linestyle=linestyle_map[ptype],
                   linewidth=1.5, label=ptype)
            for ptype in parcelTypes if ptype in linestyle_map
        ]
    
        fig.legend(
            handles=custom_lines,
            loc=loc,
            ncol=len(custom_lines),
            fontsize=10,
            title='Parcel Types',
            title_fontsize=12,
            bbox_to_anchor=bbox,
            borderaxespad=0,
            frameon=True
        )

    @staticmethod
    def AddDepthLegend(axis, depths=["ALL", "SHALLOW", "DEEP"]):
        """
        Adds a legend to a specific axis for cloud depth categories (color-coded).
        """
        color_map = {
            "SHALLOW": "green",
            "DEEP": "blue"
        }
    
        legend_lines = [
            Line2D([0], [0], color=color_map[d], linestyle='solid',
                   linewidth=2, label=d)
            for d in depths if d in color_map
        ]
    
        axis.legend(
            handles=legend_lines,
            loc='upper right',
            title='Cloud Types',
            title_fontsize=10,
            fontsize=9,
            frameon=True
        )
    
    # === Level 3: Plot one line ===
    @staticmethod
    def PlotProfileLine(axis, profile, SE_profile, parcelType, parcelDepth, multiplier=1):
        avg = TrackedProfiles_Plotting_CLASS.ProfileMean(profile)
        x = multiplier * avg[:, 0]
        y = avg[:, 1]
    
        color = TrackedProfiles_Plotting_CLASS.depth_colors.get(parcelDepth, "gray")
        linestyle = TrackedProfiles_Plotting_CLASS.category_styles.get(parcelType, "solid")
        label = f"{parcelType}-{parcelDepth}"
    
        # Plot main line
        axis.plot(x, y, color=color, linestyle=linestyle, linewidth=1, label=label)
    
        # Plot SE band
        if SE_profile is not None:
            TrackedProfiles_Plotting_CLASS.PlotSE(axis, avg, SE_profile, color=color, multiplier=multiplier)
    
    # === Level 2: Plot all depths for a given parcelType ===
    @staticmethod
    def PlotAllDepths(axis, profiles, profilesSE, parcelType, variableName, parcelDepths, multiplier=1, zlim=(0,6)):
        for parcelDepth in parcelDepths:
            profile = profiles[parcelType][parcelDepth][variableName]["profile_array"]
            SE_profile = None
            if profilesSE:
                SE_profile = profilesSE[parcelType][parcelDepth][variableName].get("profile_array_SE")
            TrackedProfiles_Plotting_CLASS.PlotProfileLine(axis, profile, SE_profile, parcelType, parcelDepth, multiplier=multiplier)
    
        TrackedProfiles_Plotting_CLASS.ApplyXLimFromZLim(axis, zlim)
            
    # === Level 1: Plot one variable to a single axis ===
    @staticmethod
    def PlotSingleVariable(axis, profiles, profilesSE, variableName, variableInfo,
                           parcelTypes, parcelDepths, hLines,hLineColors):
        label = variableInfo[variableName]["label"]
        units = variableInfo[variableName]["units"]
        multiplier = variableInfo[variableName].get("multiplier", 1)
    
        for parcelType in parcelTypes:
            
            TrackedProfiles_Plotting_CLASS.PlotAllDepths(axis, profiles, profilesSE, parcelType, variableName, parcelDepths, multiplier=multiplier)
            if variableName in ['VMF_g']:
                TrackedProfiles_Plotting_CLASS.PlotAllDepths(axis, profiles, profilesSE, parcelType, "VMF_c", parcelDepths, multiplier=multiplier)
    
        axis.set_ylabel("Height (km)")
        axis.set_xlabel(f"{label} {units}")
        axis.grid(True, linestyle="--", alpha=0.4)
        TrackedProfiles_Plotting_CLASS.PlotHLines(axis, hLines, hLineColors)
    
    # === Top Level: Make Figure and Plot all variables ===
    # Need custom made function for each plot type

