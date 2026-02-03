import numpy as np
import h5py, torch, logging 
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def save_hdf5(X: np.ndarray, y_counts: np.ndarray, filepath: str,
                number_peaks: int, sequence_length: int,  y_prof: np.ndarray = None): 
    
    """
    Save one-hot encoded sequences and prediction targets to an HDF5 file.

    This function validates the shapes of the provided arrays representing
    genomic input sequences and associated targets (counts and optionally profiles),
    and then writes them into an HDF5 file. Each array is stored as a separate
    dataset using 64-bit floating point precision. A log message is printed
    indicating the file being saved.

    Parameters
    ----------
    X : np.ndarray
        Input one-hot encoded DNA sequences of shape 
        `(number_peaks, 4, sequence_length)`.
    y_counts : np.ndarray
        Target count values per sample, of shape `(number_peaks, 1)`.
    filepath : str
        Path to the output HDF5 file where data will be saved.
    number_peaks : int
        Number of samples (peaks) expected in the dataset.
    sequence_length : int
        Length of each input DNA sequence.
    y_prof : np.ndarray, optional
        Optional per-base signal profiles of shape `(number_peaks, sequence_length)`.
        If provided, it will be saved under the dataset `'data_y_prof'`.
    """
    assert X.shape == (number_peaks, 4, sequence_length), "Double check X tensor shape, should be SAMPLESx4xSEQ_LENGTH"
    assert y_counts.shape == (number_peaks, 1), "Double check y_counts tensor shape, should be SAMPLESx1"
    if not y_prof is None:
        assert y_prof.shape == (number_peaks, sequence_length), "Double check y_prof tensor shape, should be SAMPLESxSEQ_LENGTH"

    logging.info(f"Saving {filepath}")

    with h5py.File(filepath, "w") as f: 
        f.create_dataset('data_X', data = torch.tensor(X), dtype = "float64")
        f.create_dataset('data_y_counts', data = torch.tensor(y_counts), dtype = "float64")
        if not y_prof is None:
            f.create_dataset('data_y_prof', data = torch.tensor(y_prof), dtype = "float64")


class BPNetDataset(Dataset): 
    """
    PyTorch Dataset for loading BPNet-style inputs and targets from an HDF5 file.

    This dataset assumes an HDF5 file containing three datasets:
    - "data_X": input tensors (e.g., one-hot encoded sequences or signal tracks) shape `(number_peaks, 4, sequence_length)`
    - "data_y_counts": scalar or vector counts targets (e.g., total read counts) shape `(number_peaks, 1)`
    - "data_y_prof": profile targets (e.g., base-resolution coverage profiles) shape `(number_peaks, sequence_length)`

    The dataset transparently reads slices from disk on each __getitem__ call and
    returns them as PyTorch tensors, suitable for use with a DataLoader.
    """


    def __init__(self, input_HDF5: str, device = None):

        if device == None: 
            self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
            self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        else: 
            self.device = device 

        self.hdf5 = h5py.File(input_HDF5, "r")

    def __len__(self): 
        return self.hdf5["data_X"].shape[0]
    
    def __getitem__(self, idx):

        tensor_x = torch.tensor(self.hdf5["data_X"][idx])
        tensor_y_counts = torch.tensor(self.hdf5["data_y_counts"][idx])
        tensor_Y_prof = torch.tensor(self.hdf5["data_y_prof"][idx])
        return tensor_x, tensor_y_counts, tensor_Y_prof


