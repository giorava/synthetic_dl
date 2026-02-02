import unittest
import pybedtools, sys
import pandas as pd
sys.path.insert(0, "../src")
import input_utils as data_utils

class test_input_utils(unittest.TestCase):

    def __init__(self, peaks_files, chrom_sizes, fasta_genome_path, ext): 
        super().__init__

        self.peaks = pybedtools.BedTool(peaks_files)
        self.chrom_sizes = pd.read_csv(chrom_sizes, sep = "\t", header = None)
        self.fasta_genome_path = fasta_genome_path
        self.extend = ext

    def assert_equal_bed(self, bed1, bed2): 

        bed1_df = bed1.to_dataframe()
        bed2_df = bed2.to_dataframe()
        pd.testing.assert_frame_equal(bed1_df, bed2_df)

    def test_find_overlaps(self): 
        
        w1 = pd.DataFrame([
            ("chr1",   1, 100),
            ("chr1",  30, 130),
            ("chr1", 200, 250),
            ("chr2",   1, 100)
        ])
        w1_post = pd.DataFrame([
            ("chr1", 200, 250),
            ("chr2",   1, 100)
        ])
        filtered_w1 = data_utils.filter_overlaps(w1)
        pd.testing.assert_frame_equal(filtered_w1, w1_post)

        w2 = pd.DataFrame([
            ("chr1",   1, 100),
            ("chr1",   1, 100),
            ("chr1",  80, 150),
            ("chr2",   1, 100), 
            ("chr2",   80, 140),
            ("chr3",   80, 140)
        ])
        w2_post = pd.DataFrame([
            ("chr3",   80, 140)
        ])        
        filtered_w2 = data_utils.filter_overlaps(w2)
        pd.testing.assert_frame_equal(filtered_w2, w2_post)

    def test_extend_and_filter_overlaps(self):

        chrom_sizes = pd.DataFrame([
            ("chr1",   10000),
            ("chr2",  10000),
        ])

        # checking that we are filtering out stuff outside ranges lower
        w1 = pybedtools.BedTool.from_dataframe(pd.DataFrame([
            ("chr1",   1, 10),
            ("chr2",  150, 600),
        ]))
        w1_post = pybedtools.BedTool.from_dataframe(pd.DataFrame([
            ("chr2",  (150+225)-25, (150+225)+25),
        ]))
        filtered_w1 = data_utils.extend_and_filter_overlaps(
            peaks=w1, chrom_sizes=chrom_sizes, window_size=50
        )
        self.assert_equal_bed(filtered_w1, w1_post)


        # checking ranges that become overlapping after extension
        w2 = pybedtools.BedTool.from_dataframe(pd.DataFrame([
            ("chr1", 400, 500),
            ("chr1", 598, 602),
            ("chr2",  150, 600),
        ]))
        w2_post = pybedtools.BedTool.from_dataframe(pd.DataFrame([
            ("chr2",  (150+225)-200, (150+225)+200),
        ]))
        filtered_w2 = data_utils.extend_and_filter_overlaps(
            peaks=w2, chrom_sizes=chrom_sizes, window_size=400
        )
        self.assert_equal_bed(filtered_w2, w2_post)


        # checking that we are filtering out stuff outside ranges upper
        w3 = pybedtools.BedTool.from_dataframe(pd.DataFrame([
            ("chr1",   10000-2, 10000),
            ("chr2",   150, 600),
        ]))
        w3_post = pybedtools.BedTool.from_dataframe(pd.DataFrame([
            ("chr2",  (150+225)-25, (150+225)+25),
        ]))

        filtered_w3 = data_utils.extend_and_filter_overlaps(
            peaks=w3, chrom_sizes=chrom_sizes, window_size=50
        )
        self.assert_equal_bed(filtered_w3, w3_post)

    def test_remove_Ns(self): 
            
        w1 = pybedtools.BedTool.from_dataframe(pd.DataFrame([
            ("chr1",   500, 600),
            ("chr2",  4999995, 5000005),
        ]))
        w1_post = pd.DataFrame([
            ("chr2",  4999995, 5000005),
        ])
        filtered_w1 = data_utils.check_for_Ns(
            peaks=w1, genome_fasta_path=self.fasta_genome_path
        )
        pd.testing.assert_frame_equal(filtered_w1, w1_post)

if __name__=="__main__": 
    
    obj = test_input_utils(peaks_files = "../demos/example_datasets/peaks.bed",
                        chrom_sizes = "/homes/users/gravanelli/reference_genomes/mm10.chrom.sizes",
                        fasta_genome_path = "/homes/users/gravanelli/reference_genomes/mm10.fa",
                        ext = 1000)
    obj.test_find_overlaps()
    obj.test_extend_and_filter_overlaps()
    obj.test_remove_Ns()
