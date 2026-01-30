import torch, tqdm
from torch import nn
import pandas as pd
import numpy as np
import pybedtools, tqdm, pyBigWig
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 

import datasets_manager, models
import matplotlib.pyplot as plt
from torchvision import transforms
import scipy, argparse, warnings


class prepareInput(): 

    def __init__(self, datadir: str, regions: str, chromsizes: str, genomefasta: str, input_length: int,
                        bigwigs: list[str], test_chrom: list[str], train_chrom: list[str], val_chrom: list[str]):

        self.datadir = datadir
        self.regions_file = regions
        self.chromsizes_file = chromsizes
        self.genomefasta_file = genomefasta
        self.bigwigs_files = bigwigs
        self.test_chrom = test_chrom
        self.train_chrom = train_chrom
        self.val_chrom = val_chrom
        self.input_length = input_length

        self.regions = pybedtools.BedTool(self.regions_file)
        self.chromsizes = pd.read_csv(self.chromsizes_file, sep = "\t", header = None)

        self.n_heads = len(bigwigs)
        self.n_tasks = len(bigwigs[0]) 

        self.bws = [[pyBigWig.open(j) for j in i] for i in bigwigs]

    def get_bin_centered(self, coords: list, ext: int):

        center = (coords[1]+coords[2])//2
        new_low, new_high = center-ext, center+ext
        size = self.chromsizes[self.chromsizes.iloc[:, 0] == coords[0]][1].to_numpy()[0]

        if (new_low>1)&(new_high<size):
            return (coords[0], new_low, new_high)
        else: 
            return None
        

    def one_hot_and_check(self, coords): 
        chrom, start, end = coords
        sequence = pybedtools.BedTool.seq((chrom, start, end), self.genomefasta_file).upper()
        count_Ns = sequence.count("N")
    
        if count_Ns == 0: 
            mapping = dict(zip("ACGT", range(4)))
            hot_enc = np.zeros((len(sequence), 4))
            hot_enc[np.arange(len(sequence)), [mapping[i] for i in sequence]] = 1
            return hot_enc
        else: 
            return None
        
        
    def bigwigs_binned(self, coords): 

        for head_idx in range(self.n_heads):
            task_signal = []
            for task_idx in range(self.n_tasks): 
                bw_values = self.bws[head_idx][task_idx].values(coords[0], start = coords[1], end = coords[2])
                task_signal.append(np.nan_to_num(bw_values))

        
    def get_signal_binned(self):

        if self.regions.shape[1]>3:
                warnings.warn("BED file with more than 3 columns, only chrom, strat and end will be considered")

        for i, region in tqdm.tqdm(self.regions):
            cordinates = region.chrom, region.start, region.end
            centered_bin = self.get_bin_centered(cordinates, ext = self.input_length//2) 
            if centered_bin != None: 
                hot_encoding = self.one_hot_and_check(cordinates)
                if hot_encoding != None: 

                    ## convert hot_encoding_to_tensor
                    ## get tensor data






        

        # for region in regions
        #       get the coordinates
        #       center the bin
        #.      if centered bin is not None:
        #           one hot encoding of the sequence
        #           if one hot encoded is not None:
        #               signal_head = []
        #               for each task in bigwigs.shape[0]:
        #                      read the 
    
        

