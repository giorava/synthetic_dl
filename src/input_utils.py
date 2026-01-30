import pandas as pd
import pybedtools 
import tqdm
import logging
import numpy as np
from pybedtools import BedTool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def filter_overlaps(extended_peaks: pd.DataFrame) -> pd.DataFrame: 
    """
    Filter DataFrame of genomic intervals by removing overlapping peaks on the same chromosome.
    
    For each interval, counts overlapping intervals on the same chromosome (any overlap 
    where another interval's start/end falls within the current interval). Removes all 
    intervals with ≥1 overlap.
    
    Args:
        extended_peaks (pd.DataFrame): BED-like DataFrame with columns 
                                       [chrom, start, end] (unnamed, 0-indexed)
    
    Returns:
        pd.DataFrame: Non-overlapping intervals with index reset
    """
    
    overlaps_array = []
    for i, row in extended_peaks.iterrows(): 
        chrom_current, start_current, end_current = row[0], row[1], row[2]
        filtered_removed = extended_peaks.drop(index = i)
        filterd_chrom = filtered_removed[filtered_removed.iloc[:, 0] == chrom_current]

        first_and = (filterd_chrom.iloc[:, 1]>=start_current)&(filterd_chrom.iloc[:, 1]<=end_current)
        second_and = (filterd_chrom.iloc[:, 2]>=start_current)&(filterd_chrom.iloc[:, 2]<=end_current)
        number_overlaps = np.sum(first_and|second_and)
        overlaps_array.append(number_overlaps.item())
        
    overlapping_indx = np.where(np.array(overlaps_array) != 0)[0]
    return extended_peaks.drop(index = overlapping_indx).reset_index()


def extend_and_filter_overlaps(peaks: BedTool, chrom_sizes: pd.DataFrame, window_size: int) -> BedTool:
    """
    Extend peaks to fixed window size around center and filter for non-overlapping regions.
    
    Centers each peak, extends symmetrically to target window_size, clips to chromosome 
    boundaries, then removes overlapping intervals using filter_overlaps().
    
    Args:
        peaks (BedTool): Input BED peaks to extend
        chrom_sizes (pd.DataFrame): Chromosome sizes with columns [chrom_name, length]
        window_size (int): Target width for extended peaks (symmetric around center)
    
    Returns:
        BedTool: Non-overlapping extended peaks as ['chrom', 'start', 'end'] intervals
    
    Raises:
        AssertionError: If chrom_sizes doesn't have exactly 2 columns (chrom id and lenght)
    """
    
    assert chrom_sizes.shape[1] != 2, "Check that the fist column of chrom sizes are chromosome names"
    chrom_sizes.index = chrom_sizes.iloc[:,0]

    logging.info(f"Extending {peaks.__len__()} peaks")
    extended_peaks = []
    for peak in tqdm.tqdm(peaks):
        chrom, start, end = peak.chrom, peak.start, peak.end
        chrom_size = chrom_sizes.loc[chrom][1].item()
        mid_peak = (end-start)//2
        lower_bound = mid_peak-window_size//2
        upper_bound = mid_peak+window_size//2

        if (lower_bound<0)|(upper_bound>chrom_size): 
            continue
        else: 
            extended_peaks.append((chrom, lower_bound, upper_bound))
    
    logging.info(f"Filtering {len(extended_peaks)} peaks for overlaps")
    extended_peaks = pd.DataFrame(extended_peaks)
    filtered = filter_overlaps(extended_peaks)
    logging.info(f"Number of submitted peaks: {peaks.__len__()}")
    logging.info(f"Number of non overlapping peaks: {filtered.shape[0]}")

    return pybedtools.BedTool.from_dataframe(filtered, from_list=True)


def check_for_Ns(peaks: BedTool, genome_fasta_path: str) -> pd.DataFrame: 
    """
    Filter peaks by removing those containing 'N' bases in their genomic sequence.
    
    Extracts the genomic sequence for each peak using pybedtools.seq(), counts 'N' 
    bases (case-insensitive), and retains only peaks with zero 'N' bases.
    
    Args:
        peaks (BedTool): Input BED intervals representing peaks to filter.
        genome_fasta_path (str): Path to reference genome FASTA file.
    
    Returns:
        pd.DataFrame: Filtered peaks with columns ['chrom', 'start', 'end'] 
                      containing only peaks without 'N' bases.
    """
    
    surviving_peaks = []
    for peak in tqdm.tqdm(peaks): 
        chrom, start, end = peak.chrom, peak.start, peak.end
        sequence = pybedtools.BedTool.seq((chrom, start, end), genome_fasta_path).upper()
        count_Ns = sequence.count("N")
        if count_Ns == 0:  
            surviving_peaks.append((chrom, start, end))
        else:
            continue

    logging.info(f"Number of submitted peaks: {peaks.__len__()}")
    logging.info(f"Number of surviving peaks: {len(surviving_peaks)}")
    
    return pd.DataFrame(surviving_peaks)

    



































# import torch, tqdm
# from torch import nn
# import pandas as pd
# import numpy as np
# import pybedtools, tqdm, pyBigWig
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader 

# import datasets_manager, models
# import matplotlib.pyplot as plt
# from torchvision import transforms
# import scipy, argparse, warnings

# class prepareInput(): 

#     def __init__(self, datadir: str, regions: str, chromsizes: str, genomefasta: str, input_length: int,
#                         bigwigs: list[str], test_chrom: list[str], train_chrom: list[str], val_chrom: list[str]):

#         self.datadir = datadir
#         self.regions_file = regions
#         self.chromsizes_file = chromsizes
#         self.genomefasta_file = genomefasta
#         self.bigwigs_files = bigwigs
#         self.test_chrom = test_chrom
#         self.train_chrom = train_chrom
#         self.val_chrom = val_chrom
#         self.input_length = input_length

#         self.regions = pybedtools.BedTool(self.regions_file)
#         self.chromsizes = pd.read_csv(self.chromsizes_file, sep = "\t", header = None)

#         self.n_heads = len(bigwigs)
#         self.n_tasks = len(bigwigs[0]) 

#         self.bws = [[pyBigWig.open(j) for j in i] for i in bigwigs]

#     def get_bin_centered(self, coords: list, ext: int):

#         center = (coords[1]+coords[2])//2
#         new_low, new_high = center-ext, center+ext
#         size = self.chromsizes[self.chromsizes.iloc[:, 0] == coords[0]][1].to_numpy()[0]

#         if (new_low>1)&(new_high<size):
#             return (coords[0], new_low, new_high)
#         else: 
#             return None
        

#     def one_hot_and_check(self, coords): 
#         chrom, start, end = coords
#         sequence = pybedtools.BedTool.seq((chrom, start, end), self.genomefasta_file).upper()
#         count_Ns = sequence.count("N")
    
#         if count_Ns == 0: 
#             mapping = dict(zip("ACGT", range(4)))
#             hot_enc = np.zeros((len(sequence), 4))
#             hot_enc[np.arange(len(sequence)), [mapping[i] for i in sequence]] = 1
#             return hot_enc
#         else: 
#             return None
        
        
#     def bigwigs_binned(self, coords): 

#         for head_idx in range(self.n_heads):
#             task_signal = []
#             for task_idx in range(self.n_tasks): 
#                 bw_values = self.bws[head_idx][task_idx].values(coords[0], start = coords[1], end = coords[2])
#                 task_signal.append(np.nan_to_num(bw_values))

        
#     def get_signal_binned(self):

#         if self.regions.shape[1]>3:
#                 warnings.warn("BED file with more than 3 columns, only chrom, strat and end will be considered")

#         for i, region in tqdm.tqdm(self.regions):
#             cordinates = region.chrom, region.start, region.end
#             centered_bin = self.get_bin_centered(cordinates, ext = self.input_length//2) 
#             if centered_bin != None: 
#                 hot_encoding = self.one_hot_and_check(cordinates)
#                 if hot_encoding != None: 

#                     ## convert hot_encoding_to_tensor
#                     ## get tensor data






        

#         # for region in regions
#         #       get the coordinates
#         #       center the bin
#         #.      if centered bin is not None:
#         #           one hot encoding of the sequence
#         #           if one hot encoded is not None:
#         #               signal_head = []
#         #               for each task in bigwigs.shape[0]:
#         #                      read the 
    
        

