#!/bin/bash
#SBATCH --job-name=python
#SBATCH --account=torri
#SBATCH --partition=torri

##SBATCH --partition=shared

#SBATCH --time=0-23:00:00
#SBATCH --nodes=1 
#SBATCH  --tasks-per-node=1  
#SBATCH --cpus-per-task=1 #Each node has 47 CPUs
#SBATCH --mem=10G #GBTotal 180GB per node (180G/30 = 6G) #0 for full node memory

#SBATCH --error=pybash_job-%A.err
#SBATCH --output=pybash_job-%A.out
## Remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=air673@hawaii.edu

mkdir -p job_out/access_files

#LOADING PYTHON MODULES
module purge
module load lang/Anaconda3
source activate work #personal python custom environment
#PYTHON SETTINGS
export HDF5_USE_FILE_LOCKING=FALSE #disable HDF5 file locking
export PYTHONUNBUFFERED=TRUE #allows print statements during run


# --- RUNNING PYTHON ---
python -u access_files.py $SLURM_ARRAY_TASK_ID \
     > job_out/access_files/py-${SLURM_ARRAY_TASK_ID}.out 2>&1
