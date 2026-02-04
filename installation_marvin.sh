#!/bin/bash

### IMPORTANT: EXECUTE THE INSTALLATION SCRIPT ONLY FROM INTERACTIVE SESSION OR SUBMIT AS BATCH

module load CUDA/12.5.0
module load Miniconda3/23.9.0-0

# ------------------------------------------------------------------------------
# Function: checkPackage
# Usage: checkPackage <package_name>
#
# Description:
#   Attempts to import a specified Python package using Python 3.
#   If the import succeeds, it prints a success message.
#   If the import fails, it prints an error message and terminates the script.
#
# Arguments:
#   package_name : Name of the Python package to check (string)
#
# Returns:
#   Exits the script with an error code if the package cannot be imported.
#
# Example:
#   checkPackage numpy
#
# ------------------------------------------------------------------------------
checkPackage() {
    python3 -c "import $1"
    errorVal=$?
    if [ $errorVal -eq 0 ]; then
        echo "$1 imported successfully."
    else
        echo "ERROR DETECTED: Failed to import $1"
        exit
    fi
}

# create enviroment with python
conda create -p $HOME/scratch/envs/synth_v2 python -y
conda activate $HOME/scratch/envs/synth_v2

# install pythorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
checkPackage torch  
checkPackage torchvision 
checkPackage numpy 

# install various packages
conda install bioconda::bedtools -y
conda install jupyter -y
pip3 install pandas matplotlib pybedtools pybigwig scipy tqdm h5py
checkPackage pandas 
checkPackage matplotlib 
checkPackage pybedtools
checkPackage pyBigWig
checkPackage scipy
checkPackage tqdm
checkPackage h5py
