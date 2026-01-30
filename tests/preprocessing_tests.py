import unittest
import pybedtools
import pandas as pd
import src.input_utils as data_utils

class test_input_utils(unittest.TestCase):

    def __init__(self, peaks_files, chrom_sizes, ext): 
        super().__init__

        self.peaks = pybedtools.BedTool(peaks_files)
        self.chrom_sizes = pd.read_csv(chrom_sizes, sep = "\t", header = None)
        self.extend = ext

    def test_find_overlaps(): 
        pass
    
    def test_extend_and_filter_overlaps():
        pass 

    def check_for_Ns():
        pass

if __name__=="__main__": 
    pass