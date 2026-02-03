import numpy as np
import h5py, torch, logging 
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def save_hdf5(X: np.ndarray, y_counts: np.ndarray, filepath: str,
                number_peaks: int, sequence_length: int, number_tasks:int,  y_prof: np.ndarray = None) -> None: 
    """
    Save BPNet-style input and target tensors to an HDF5 file with task-specific datasets.

    Validates tensor shapes and creates separate datasets for each task's counts and profile
    targets. Designed for multi-task BPNet models where each task has both count (scalar)
    and profile (sequence-level) predictions.

    Args:
        X (np.ndarray): Input features of shape (number_peaks, 4, sequence_length).
                        Typically one-hot encoded sequences (A/C/G/T channels).
        y_counts (np.ndarray): Count targets of shape (number_tasks, number_peaks, 1).
                              Scalar read counts per peak per task.
        y_prof (np.ndarray | None): Profile targets of shape 
                                   (number_tasks, number_peaks, sequence_length).
                                   Optional base-resolution profile targets.
        filepath (str): Output HDF5 file path.
        number_peaks (int): Expected number of peaks/samples.
        number_tasks (int): Number of prediction tasks/models.
        sequence_length (int): Expected sequence length/window size.

    Creates datasets:
        - 'data_X': Input tensor (float64)
        - 'data_y_counts_task_{i}': Counts for task i (float64)
        - 'data_y_prof_task_{i}': Profile for task i (float64, if y_prof provided)
    """
    
    assert X.shape == (number_peaks, 4, sequence_length), "Double check X tensor shape, should be SAMPLESx4xSEQ_LENGTH"
    assert y_counts.shape == (number_tasks, number_peaks, 1), "Double check y_counts tensor shape, should be SAMPLESx1"
    if not y_prof is None:
        assert y_prof.shape == (number_tasks, number_peaks, sequence_length), "Double check y_prof tensor shape, should be SAMPLESxSEQ_LENGTH"

    logging.info(f"Saving {filepath}")

    with h5py.File(filepath, "w") as f: 
        f.create_dataset('data_X', data = torch.tensor(X), dtype = "float64")
        for i in range(number_tasks):
            f.create_dataset(f'data_y_counts_task_{i}', data = torch.tensor(y_counts[i, :, :]), dtype = "float64")
        if not y_prof is None:
            for i in range(number_tasks):
                f.create_dataset(f'data_y_prof_task_{i}', data = torch.tensor(y_prof[i, :, :]), dtype = "float64")


class BPNetDataset(Dataset): 
    """
    PyTorch Dataset for loading BPNet-style inputs and targets from an HDF5 file.

    This dataset assumes an HDF5 file containing three datasets:
    - "data_X": input tensors (e.g., one-hot encoded sequences or signal tracks) shape `(number_peaks, 4, sequence_length)`
    - "data_y_counts_task_ID": scalar or vector counts targets (e.g., total read counts) shape `(number_peaks, 1)`
    - "data_y_prof_task_ID": profile targets (e.g., base-resolution coverage profiles) shape `(number_peaks, sequence_length)`

    The dataset transparently reads slices from disk on each __getitem__ call and
    returns them as PyTorch tensors, suitable for use with a DataLoader.
    """


    def __init__(self, input_HDF5: str, number_tasks:int, device = None):

        if device == None: 
            self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
            self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        else: 
            self.device = device 

        self.hdf5 = h5py.File(input_HDF5, "r")
        self.number_tasks = number_tasks

    def __len__(self): 
        return self.hdf5["data_X"].shape[0]
    
    def __getitem__(self, idx):

        output = []
        output += [torch.tensor(self.hdf5["data_X"][idx])]
        for task in range(self.number_tasks):
            output += [torch.tensor(self.hdf5[f"data_y_prof_task_{task}"][idx])]
            output += [torch.tensor(self.hdf5[f"data_y_counts_task_{task}"][idx])]

        return output


