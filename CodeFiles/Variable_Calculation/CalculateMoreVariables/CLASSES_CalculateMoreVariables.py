#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# ModelData_Class
# ============================================================

#Libraries
import os
import xarray as xr
from datetime import timedelta

class ModelData_Class:
    def __init__(self, mainDirectory, scratchDirectory, simulationNumber):
        self.mainDirectory = mainDirectory
        self.scratchDirectory = scratchDirectory
        self.simulationNumber = simulationNumber
        
        # Initialize directories and metadata
        (self.dataDirectory, 
         self.parcelDirectory, 
         self.res, 
         self.t_res, 
         self.Np_str, 
         self.Nz_str) = self.GetDataDirectories()

        # Load coordinate data only (lightweight)
        self.GetCoordinateData()
        self.timeStrings = self.GetTimeStrings(self.time)

        # Print summary
        self.Summary()

    # ============================================================
    # ========== Data Loading Functions ==========
    # ============================================================

    def GetDataDirectories(self):
        """Return directory paths and metadata based on simulation number."""
        if self.simulationNumber == 1:
            Directory = os.path.join(self.mainDirectory, 'Model/cm1r20.3/run')
            res, t_res, Np_str, Nz_str = '1km', '5min', '1e6', '34'
        elif self.simulationNumber == 2:
            Directory = self.scratchDirectory
            res, t_res, Np_str, Nz_str = '1km', '1min', '50e6', '95'
        elif self.simulationNumber == 3:
            Directory = self.scratchDirectory
            res, t_res, Np_str, Nz_str = '250m', '1min', '50e6', '95'
        else:
            raise ValueError("Invalid simulationNumber (must be 1, 2, or 3).")

        dataDirectory = os.path.join(Directory, f"cm1out_{res}_{t_res}_{Nz_str}nz.nc")
        parcelDirectory = os.path.join(Directory, f"cm1out_pdata_{res}_{t_res}_{Np_str}np.nc")
        return dataDirectory, parcelDirectory, res, t_res, Np_str, Nz_str


    def GetCoordinateData(self):
        """
        Extract coordinate arrays (time, zf, zh, yf, yh, xf, xh) 
        from the CM1 dataset and immediately close the file.
        """
        with xr.open_dataset(self.dataDirectory, decode_timedelta=True) as ds:
            coords = ['time', 'zf', 'zh', 'yf', 'yh', 'xf', 'xh']
            extracted = {k: ds[k].values for k in coords}

        # Assign coordinate arrays and their lengths
        for k, v in extracted.items():
            setattr(self, k, v)
            setattr(self, f"N{k}", len(v))
            
        return extracted


    def GetTimeStrings(self, times):
        """Convert CM1 time array (nanoseconds) to formatted strings."""
        return [str(timedelta(seconds=float(s))).replace(":", "-") for s in times / 1e9]


    # ============================================================
    # ========== On-demand Variable Access ==========
    # ============================================================

    def OpenDataset(self, decode_timedelta=True):
        #EXAMPLE: ds = ModelData.OpenDataset()
        #         ...
        #         del ds
        """
        Permanently open the NetCDF dataset (kept open until manually closed).

        Use this when you need persistent access for multiple operations.
        """
        if hasattr(self, "ds") and self.ds is not None:
            print("Dataset already open.")
            return self.ds

        self.ds = xr.open_dataset(self.dataDirectory, decode_timedelta=decode_timedelta)
        print(f"✅ Opened dataset: {self.dataDirectory}")
        return self.ds

    def GetVariable(self, varName, isel=None):
        #EXAMPLE: w = ModelData.GetVariable('winterp', isel={'time': slice(0,2), 'zh': 0, 'yh': 0, 'xh': 0}) #example getting a variable
        """
        Open the full NetCDF file, extract a variable (optionally subset via .isel), 
        then close immediately. Returns the variable data as a NumPy array.

        Parameters
        ----------
        varName : str
            Name of the variable to extract.
        isel : dict, optional
            Dictionary of indices to select (e.g., {'time': 0, 'zh': slice(0,10)}).
        decode_timedelta : bool, optional
            Whether to decode CF-style timedelta coordinates (default: True).
        """
        with xr.open_dataset(self.dataDirectory, decode_timedelta=True) as ds:
            if varName not in ds.variables:
                raise KeyError(f"Variable '{varName}' not found in dataset.")
            da = ds[varName]
            if isel is not None:
                da = da.isel(**isel)
            varData = da.data  # load into memory before closing
        return varData

    # ============================================================
    # === Information ========================================
    # ============================================================

    def Summary(self):
        """Print a summary of the simulation configuration."""
        print("=== CM1 Data Summary ===")
        print(f" Simulation #:   {self.simulationNumber}")
        print(f" Resolution:     {self.res}")
        print(f" Time step:      {self.t_res}")
        print(f" Vertical levels:{self.Nz_str}")
        print(f" Parcels:        {self.Np_str}")
        print(f" Data file:      {self.dataDirectory}")
        print(f" Parcel file:    {self.parcelDirectory}")
        print(f" Time steps:     {len(self.time)}")
        print("=========================")

#EXAMPLE: ModelData = ModelData_Class(mainDirectory, scratchDirectory, simulationNumber=1)


# In[2]:


# ============================================================
# DataManager_Class
# ============================================================

#Libraries
import os
import h5py

class DataManager_Class:
    def __init__(self, mainDirectory, scratchDirectory, res, t_res, Nz_str, dataName, dtype):
        self.mainDirectory = mainDirectory
        self.scratchDirectory = scratchDirectory
        self.res = res
        self.t_res = t_res
        self.Nz_str = Nz_str
        self.dataName = dataName
        self.dtype = dtype

        # Initialize directories on creation
        self.inputDirectory = self.GetInputDirectory(mainDirectory, scratchDirectory, res)
        self.outputDirectory = self.GetOutputDirectory(mainDirectory, scratchDirectory, res)
        self.inputDataDirectory = self.MakeInputDataDirectory(self.inputDirectory, res, t_res, Nz_str)
        self.outputDataDirectory = self.MakeOutputDataDirectory(self.outputDirectory, res, t_res, Nz_str)

    # ============================================================
    # ========== Functions ==========
    # ============================================================

    def GetInputDirectory(self, mainDirectory, scratchDirectory, res):
        if res == '1km':
            inputDirectory = os.path.join(mainDirectory, 'Code', 'OUTPUT', 'Variable_Calculation', 'TimeSplitModelData')
        if res == '250m':
            inputDirectory = os.path.join(scratchDirectory, 'OUTPUT', 'Variable_Calculation', 'TimeSplitModelData')
        return inputDirectory

    def GetOutputDirectory(self, mainDirectory, scratchDirectory, res):
        if res == '1km':
            outputDirectory = os.path.join(mainDirectory, 'Code', 'OUTPUT', 'Variable_Calculation', 'CalculateMoreVariables')
            os.makedirs(outputDirectory, exist_ok=True)
        if res == '250m':
            outputDirectory = os.path.join(scratchDirectory, 'OUTPUT', 'Variable_Calculation', 'CalculateMoreVariables')
            os.makedirs(outputDirectory, exist_ok=True)
        return outputDirectory

    def MakeInputDataDirectory(self, inputDirectory, res, t_res, Nz_str):
        inputDataDirectory = os.path.join(inputDirectory, f"{res}_{t_res}_{Nz_str}nz", "ModelData")
        return inputDataDirectory

    def MakeOutputDataDirectory(self, outputDirectory, res, t_res, Nz_str):
        outputDataDirectory = os.path.join(outputDirectory, f"{res}_{t_res}_{Nz_str}nz",self.dataName)
        os.makedirs(outputDataDirectory, exist_ok=True)
        return outputDataDirectory

    def GetTimestepData(self, inputDataDirectory, timeString, VariableName):
        inputDataFile = os.path.join(
            inputDataDirectory,
            f"cm1out_{self.res}_{self.t_res}_{self.Nz_str}nz_{timeString}.h5"
        )
        with h5py.File(inputDataFile, 'r') as f:
            InputData = f[VariableName][:]
        return InputData

    def SaveOutputTimestep(self, outputDataDirectory, res, t_res, timeString, outputDictionary):
        out_file = os.path.join(
            outputDataDirectory,
            f"{self.dataName}_{res}_{t_res}_{self.Nz_str}nz_{timeString}.h5"
        )
        with h5py.File(out_file, 'w') as f:
            for var_name, arr in outputDictionary.items():
                f.create_dataset(var_name, data=arr, dtype=self.dtype, compression="gzip")
        print(f"Saved timestep to output file: {out_file}")

# EXAMPLE: DataManager = DataManager_Class(mainDirectory, scratchDirectory, ModelData.res, ModelData.t_res, ModelData.Nz_str, dataName="Eulerian_Binary_Array", dtype='bool')


# In[ ]:




