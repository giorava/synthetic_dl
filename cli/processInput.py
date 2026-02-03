import sys
import argparse
import logging
sys.path.insert(0, "src")
import input_utils 
import data_managment_utils as data_utils
import pandas as pd
import pybedtools, pyBigWig
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_arguments():

    parser = argparse.ArgumentParser(description="Create summary peaks statistics dataframe")
    parser.add_argument("--peaks", help="BED file with the peaks")
    parser.add_argument("--chrom_sizes", help="BED file with chromosome sizes")
    parser.add_argument("--genome_fasta", help="genome fasta file")
    parser.add_argument("--output_size", help="output size", type=int)
    parser.add_argument("--input_size", help="input size", type=int)
    parser.add_argument("--pos_bigwig", help="bigwig file for the positive strand")
    parser.add_argument("--neg_bigwig", help="bigwig file for the negative strand")
    parser.add_argument("--trainset", help="training chromosomes")
    parser.add_argument("--testset", help="test chromosomes")
    parser.add_argument("--valset", help="validation chromosomes")
    parser.add_argument("--output_path", help="output data path")
    args = parser.parse_args()

    return args

def save_files(peaks, ohe, bigwig_pos, bigwig_neg, output_length, path): 

    profile_pos = input_utils.get_profiles(peaks = peaks, bigwig = bigwig_pos)
    profile_neg = input_utils.get_profiles(peaks = peaks, bigwig = bigwig_neg)

    profiles = np.array([profile_pos, profile_neg])
    counts = profiles.sum(2).reshape(2, -1, 1)

    data_utils.save_hdf5(X = ohe,
                         y_counts = counts, 
                         filepath = path,
                         number_peaks = peaks.shape[0], 
                         sequence_length = output_length, 
                         number_tasks = 2,  
                         y_prof = profiles)

def run(): 

    args = parse_arguments()
    chrom_sizes = pd.read_csv(args.chrom_sizes, sep = "\t", header = None)
    peaks_bed = pybedtools.BedTool(args.peaks)   

    logging.info(f"Extending {peaks_bed.__len__()} peaks")
    filtered_peaks = input_utils.extend_and_filter_overlaps(peaks = peaks_bed, chrom_sizes = chrom_sizes, window_size = args.input_size)
    logging.info(f"Peaks surviving extension and overlap filtering {filtered_peaks.__len__()}")
    filtered_peaks = input_utils.check_for_Ns(peaks = filtered_peaks, genome_fasta_path = args.genome_fasta)
    logging.info(f"Peaks surviving Ns filtering {filtered_peaks.__len__()}")

    logging.info(f"Splitting sets and saving")
    filtered_peaks.to_csv(f"{args.output_path}/all_regions.bed", sep="\t", header=False, index=False)
    train_peaks = filtered_peaks[filtered_peaks.iloc[:,0].isin(args.trainset.split(","))]
    test_peaks = filtered_peaks[filtered_peaks.iloc[:,0].isin(args.testset.split(","))]
    val_peaks = filtered_peaks[filtered_peaks.iloc[:,0].isin(args.valset.split(","))]
    train_peaks.to_csv(f"{args.output_path}/train_regions.bed", sep="\t", header=False, index=False)
    test_peaks.to_csv(f"{args.output_path}/test_regions.bed", sep="\t", header=False, index=False)
    val_peaks.to_csv(f"{args.output_path}/val_regions.bed", sep="\t", header=False, index=False)
    
    logging.info(f"One Hot Encoding")
    train_ohe = input_utils.one_hot_encoding(peaks = train_peaks, genome_fasta_path = args.genome_fasta,  alphabet = "ACGT") 
    test_ohe = input_utils.one_hot_encoding(peaks = test_peaks, genome_fasta_path = args.genome_fasta,  alphabet = "ACGT")
    val_ohe = input_utils.one_hot_encoding(peaks = val_peaks, genome_fasta_path = args.genome_fasta,  alphabet = "ACGT")

    # train_prof_pos = get_profiles(peaks: pd.DataFrame, bigwig: pyBigWig.pyBigWig)
    # train_prof_neg = 
    bw_pos = pyBigWig.open(args.pos_bigwig)
    bw_neg = pyBigWig.open(args.neg_bigwig)

    logging.info(f"Extracting profiles")
    save_files(peaks = train_peaks, ohe = train_ohe, 
               bigwig_pos = bw_pos, bigwig_neg = bw_neg, output_length = args.output_size,
                path = f"{args.output_path}/train.h5")
    save_files(peaks = test_peaks, ohe = test_ohe, 
               bigwig_pos = bw_pos, bigwig_neg = bw_neg, output_length = args.output_size,
                path = f"{args.output_path}/test.h5")
    save_files(peaks = val_peaks, ohe = val_ohe, 
               bigwig_pos = bw_pos, bigwig_neg = bw_neg, output_length = args.output_size,
                path = f"{args.output_path}/val.h5")

if __name__=="__main__": 

    run()


