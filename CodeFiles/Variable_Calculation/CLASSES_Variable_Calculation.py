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

        # Load Variable Names
        self.varList = self.GetVariableNames()

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

    def GetVariableNames(self):
        """Get list of variable names available in the CM1 dataset."""
        with xr.open_dataset(self.dataDirectory, decode_timedelta=True) as ds:
            varList = list(ds.data_vars)
        return varList
    
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
        print(f"Opened dataset: {self.dataDirectory}")
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
        print("=========================","\n")

#EXAMPLE: ModelData = ModelData_Class(mainDirectory, scratchDirectory, simulationNumber=1)


# In[3]:


# ============================================================
# SlurmJobArray_Class
# ============================================================

import os
import numpy as np

class SlurmJobArray_Class:
    def __init__(self, total_elements, num_jobs, UsingJobArray):
        self.total_elements = total_elements
        self.num_jobs = num_jobs
        self.UsingJobArray = UsingJobArray
        
        # Get job ID (default = 1 if not running under Slurm)
        self.job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        if self.job_id == 0:
            self.job_id = 1
        
        # Precompute range info
        self.job_range = total_elements // num_jobs
        self.remaining = total_elements % num_jobs
        
        # Compute job range for this job
        self.start_job, self.end_job = self._get_job_range(self.job_id)

        # Print summary
        self.Summary()

    # ------------------------------------------------------------
    def _get_job_range(self, job_id):
        if self.UsingJobArray == True:
            """Compute start and end indices for this job."""
            job_id -= 1
            start_job = job_id * self.job_range + min(job_id, self.remaining)
            end_job = start_job + self.job_range + (1 if job_id < self.remaining else 0)
            if job_id == self.num_jobs - 1:
                end_job = self.total_elements
        elif self.UsingJobArray == False:
            start_job, end_job = 0, self.total_elements
        return start_job, end_job

    # ------------------------------------------------------------
    def TESTING(self):
        """Print start/end for all jobs to verify chunking logic."""
        start, end = [], []
        for job_id in range(1, self.num_jobs + 1):
            s, e = self._get_job_range(job_id)
            print(f"Job {job_id}: {s} → {e}")
            start.append(s)
            end.append(e)
        print("Unique starts:", len(np.unique(start)) == len(start))
        print("Unique ends:", len(np.unique(end)) == len(end))
        print("No zero-length ranges:", np.all(np.array(start) != np.array(end)))

    def Summary(self):
        print(f"Running timesteps from {self.start_job}:{self.end_job}","\n")


# In[2]:


# ============================================================
# DataManager_Class
# ============================================================

#Libraries
import os
import h5py

class DataManager_Class:
    def __init__(self, mainDirectory, scratchDirectory, res, t_res, Nz_str, Np_str, dataType, dataName, dtype):
        self.mainDirectory = mainDirectory
        self.scratchDirectory = scratchDirectory
        self.dataType = dataType
        self.res = res
        self.t_res = t_res
        self.Nz_str = Nz_str
        self.Np_str = Np_str
        self.dataName = dataName
        self.dtype = dtype

        # Initialize directories on creation
        self.inputDirectory = self.GetInputDirectory(mainDirectory, scratchDirectory)
        self.outputDirectory = self.GetOutputDirectory(mainDirectory, scratchDirectory)
        self.inputDataDirectory = self.MakeInputDataDirectory(self.inputDirectory)
        self.inputParcelDirectory = self.MakeInputParcelDirectory(self.inputDirectory)
        self.outputDataDirectory = self.MakeOutputDataDirectory(self.outputDirectory)

        # Print summary
        self.Summary()

    # ============================================================
    # ========== Functions ==========
    # ============================================================

    def GetInputDirectory(self, mainDirectory, scratchDirectory):
        if self.res == '1km':
            inputDirectory = os.path.join(mainDirectory, 'Code', 'OUTPUT', 'Variable_Calculation', 'TimeSplitModelData')
        if self.res == '250m':
            inputDirectory = os.path.join(scratchDirectory, 'OUTPUT', 'Variable_Calculation', 'TimeSplitModelData')
        return inputDirectory

    def GetOutputDirectory(self, mainDirectory, scratchDirectory):
        if self.res == '1km':
            outputDirectory = os.path.join(mainDirectory, 'Code', 'OUTPUT', 'Variable_Calculation', self.dataType)
            os.makedirs(outputDirectory, exist_ok=True)
        if self.res == '250m':
            outputDirectory = os.path.join(scratchDirectory, 'OUTPUT', 'Variable_Calculation', self.dataType)
            os.makedirs(outputDirectory, exist_ok=True)
        return outputDirectory

    def MakeInputDataDirectory(self, inputDirectory):
        inputDataDirectory = os.path.join(inputDirectory, f"{self.res}_{self.t_res}_{self.Nz_str}nz", "ModelData")
        return inputDataDirectory

    def MakeInputParcelDirectory(self, inputDirectory):
        inputParcelDirectory = os.path.join(inputDirectory, f"{self.res}_{self.t_res}_{self.Nz_str}nz", "ParcelData")
        return inputParcelDirectory

    def MakeOutputDataDirectory(self, outputDirectory):
        outputDataDirectory = os.path.join(outputDirectory, f"{self.res}_{self.t_res}_{self.Nz_str}nz",self.dataName)
        os.makedirs(outputDataDirectory, exist_ok=True)
        return outputDataDirectory

    def GetTimestepData(self, inputDataDirectory, timeString, variableName, dataName="cm1out"):
        inputDataFile = os.path.join(
            inputDataDirectory,
            f"{dataName}_{self.res}_{self.t_res}_{self.Nz_str}nz_{timeString}.h5"
        )
        with h5py.File(inputDataFile, 'r') as f:
            InputData = f[variableName][:]
        return InputData

    def GetTimestepParcel(self, inputParcelDirectory, timeString, variableName, dataName="cm1out_pdata"):
        inputDataFile = os.path.join(
            inputParcelDirectory,
            f"{dataName}_{self.res}_{self.t_res}_{self.Np_str}np_{timeString}.h5"
        )
        with h5py.File(inputDataFile, 'r') as f:
            InputData = f[variableName][:]
        return InputData

    def SaveOutputTimestep(self, outputDataDirectory, timeString, outputDictionary, dataName=None):
        if dataName is None: #allows for custom setting of dataName (set dataName to something other than None)
            dataName = self.dataName
        
        out_file = os.path.join(
            outputDataDirectory,
            f"{dataName}_{self.res}_{self.t_res}_{self.Nz_str}nz_{timeString}.h5"
        )
        with h5py.File(out_file, 'w') as f:
            for var_name, arr in outputDictionary.items():
                f.create_dataset(var_name, data=arr, dtype=self.dtype, compression="gzip")
        print(f"Saved timestep to output file: {out_file}","\n")

    def Summary(self):
        """Print a summary of the simulation configuration."""
        print("=== DataManager Summary ===")
        print(f" inputDirectory #:   {self.inputDirectory}")
        print(f" outputDirectory #:   {self.outputDirectory}")
        print(f" inputDataDirectory #:   {self.inputDataDirectory}")
        print(f" inputParcelDirectory #:   {self.inputParcelDirectory}")
        print(f" outputDataDirectory #:   {self.outputDataDirectory}")
        print("=========================","\n")

# EXAMPLE: DataManager = DataManager_Class(mainDirectory, scratchDirectory, ModelData.res, ModelData.t_res, ModelData.Nz_str, dataName="Eulerian_Binary_Array", dtype='bool')

