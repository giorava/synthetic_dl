import pandas as pd
import pybedtools 
import pyBigWig
import tqdm
import numpy as np
from pybedtools import BedTool


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
    for i, row in tqdm.tqdm(extended_peaks.iterrows()): 
        chrom_current, start_current, end_current = row[0], row[1], row[2]
        filtered_removed = extended_peaks.drop(index = i)
        filterd_chrom = filtered_removed[filtered_removed.iloc[:, 0] == chrom_current]

        first_and = (filterd_chrom.iloc[:, 1]>start_current)&(filterd_chrom.iloc[:, 1]<end_current)
        second_and = (filterd_chrom.iloc[:, 2]>start_current)&(filterd_chrom.iloc[:, 2]<end_current)
        number_overlaps = np.sum(first_and|second_and)
        overlaps_array.append(number_overlaps.item())
        
    overlapping_indx = np.where(np.array(overlaps_array) != 0)[0]

    return extended_peaks.drop(index = overlapping_indx).reset_index(drop=True)


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
    assert chrom_sizes.shape[1] == 2, "Check that the fist column of chrom sizes are chromosome names"
    chrom_sizes.index = chrom_sizes.iloc[:,0]

    extended_peaks = []
    for peak in tqdm.tqdm(peaks):
        chrom, start, end = peak.chrom, peak.start, peak.end
        chrom_size = chrom_sizes.loc[chrom][1].item()
        mid_peak = start + (end-start)//2
        lower_bound = mid_peak-window_size//2
        upper_bound = mid_peak+window_size//2

        if (lower_bound<0)|(upper_bound>chrom_size): 
            continue
        else: 
            extended_peaks.append((chrom, lower_bound, upper_bound))
    
    extended_peaks = pd.DataFrame(extended_peaks)
    filtered = filter_overlaps(extended_peaks)

    return pybedtools.BedTool.from_dataframe(filtered)


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
    
    return pd.DataFrame(surviving_peaks)

    

def get_profiles(peaks: pd.DataFrame, bigwig: pyBigWig.pyBigWig, chrom_sizes: pd.DataFrame, output_size: int) -> np.ndarray: 
    """
    Extract per-base signal profiles from a bigWig track for each peak region.

    Iterates over all peaks in the provided BedTool, retrieves the corresponding
    bigWig signal for each [chrom, start, end] interval, replaces NaNs with 0,
    and stacks the resulting profiles into a 2D NumPy array. Checks that all
    peaks have the same window size before returning.

    Args:
        peaks (BedTool): Genomic intervals (e.g., ChIP-seq peaks) for which
            signal profiles will be extracted.
        bigwig (pyBigWig.pyBigWig): Open pyBigWig handle pointing to the
            bigWig file containing the signal track.

    Returns:
        np.ndarray: Array of shape (n_peaks, window_size) with per-base signal
        values for each peak, with NaNs replaced by 0.0.
    """


    peaks = extend_and_filter_overlaps(peaks = pybedtools.BedTool.from_dataframe(peaks),
                                       chrom_sizes = chrom_sizes, window_size = output_size)

    signals = []
    w_sizes = []
    for peak in tqdm.tqdm(peaks):
        chrom, start, end = peak.chrom, peak.start, peak.end 
        vals = np.nan_to_num(bigwig.values(chrom, start, end), nan=0.0)
        signals.append(vals)
        w_sizes.append(end-start)

    assert len(np.unique(w_sizes)) == 1, "Bed file window sizes are different"
    return np.array(signals)


def one_hot_encoding(peaks: pd.DataFrame, genome_fasta_path: str,  alphabet: str = "ACGT"):
    """
    Convert genomic sequences under peaks to one-hot encoded nucleotide arrays.
    
    Extracts DNA sequences for each peak using pybedtools.BedTool.seq(), converts
    to uppercase, and creates one-hot encoding using the specified alphabet.
    Assumes A=0, C=1, G=2, T=3 mapping order by default. Validates that all peaks have
    identical window sizes before returning.
    
    Args:
        peaks (pd.DataFrame): Peak coordinates with columns [chrom, start, end]
                              (unnamed/indexed by position).
        genome_fasta_path (str): Path to reference genome FASTA file.
        alphabet (str): Nucleotide alphabet for encoding. Default: "ACGT".
    
    Returns:
        np.ndarray: One-hot encoded array of shape (n_peaks, window_size, 4)
                    where axis=2 represents A/C/G/T channels.
    """

    mapping = dict(zip(alphabet, range(4)))

    encodings = []
    w_sizes = []
    for _, peak in tqdm.tqdm(peaks.iterrows()): 
        chrom, start, end = peak[0], peak[1], peak[2]
        seq = pybedtools.BedTool.seq((chrom, start, end), genome_fasta_path).upper()
        hot_enc = np.zeros((len(seq), 4))
        hot_enc[np.arange(len(seq)), [mapping[i] for i in seq]] = 1
        encodings.append(hot_enc.T)
        w_sizes.append(end-start)

    assert len(np.unique(w_sizes)) == 1, "Bed file window sizes are different"
    return np.array(encodings)

def getLengthDifference(numDilLayers: int, initialConvolutionWidths: list[int],
                        profileKernelSize: int, verbose: bool) -> int:
    """Determine the padding on each size of the output.

    Given a BPNet architecture, calculate how much longer the input sequence will be than
    the predicted profile.

    :param numDilLayers: The number of dilated convolutional layers in BPNet.
    :param initialConvolutionWidths: The widths of the convolutional kernels preceding the
        dilated convolutions. In the original BPNet, this was a single layer of width
        25, so this argument would be [25].
    :param profileKernelSize: The width of the final kernel that generates the profiles,
        typically around 75.
    :param verbose: Should the code tell you what it's doing?
    :return: An integer representing the number of extra bases in the input compared to
        the length of the predicted profile. Divide by two to get the overhang on each
        side of the input.
    """
    overhang = 0
    for convWidth in initialConvolutionWidths:

        # First, remove the conv1 layer. The layer width must be odd.
        assert convWidth % 2 == 1
        # How many bases to trim? Consider a filter of width 5.
        #    DATADATADATADATADATA
        #    12345          12345
        #    ''PREDICTIONPREDIC''
        # We remove convWidth // 2 bases on each side, for a total of convWidth-1.
        overhang += (convWidth - 1)
        if verbose:
            print(f"Convolutional layer, width {convWidth}, receptive field {overhang + 1}")

    # Now we iterate through the dilated convolutions. The dilation rate starts at 2, then doubles
    # at each layer in the network.
    #     DATADATADATADATADATADATADATA
    #     C O N                  C O N
    #     ''INTERMEDIATEINTERMEDIATE''
    #       C   O   N      C   O   N
    #       ''''PREDICTIONPREDIC''''
    # The pattern is that the first layer removes four bases, the next eight bases, and so on.
    # N
    # __
    # \
    # /_  2^(i+1)  = 2^(i+2)-4
    # i=1
    overhang += 2 ** (numDilLayers + 2) - 4
    if verbose:
        print(f"After dilated convolutions, receptive field {overhang + 1}")
    # Now, at the bottom, we have the output filter. It's the same math as the first filter.
    assert profileKernelSize % 2 == 1
    overhang += (profileKernelSize - 1)
    if verbose:
        print(f"After final convolution, receptive field {overhang + 1}")
    return overhang

def getInputLength(outPredLen: int, numDilLayers: int, initialConvolutionWidths: list[int],
                   verbose: bool) -> int:
    """Given an output length, calculate input length.

    Given a BPNet architecture and a length of the output profile, calculate the length
    of the input sequence necessary to get that profile..

    :param outPredLen: : The length of the output profile, in bp.
    :param numDilLayers: : The number of dilated convolutional layers in BPNet.
    :param initialConvolutionWidths: : The widths of the convolutional kernels preceding
        the dilated convolutions. In the original BPNet, this was a single layer of
        width 25, so this argument would be [25].
    :param profileKernelSize: : The width of the final kernel that generates the profiles,
        typically around 75.
    :param verbose: Should the code tell you what it's doing?
    :return: An integer representing the length of the sequence necessary to calculate the profile.
    """
    return outPredLen \
        + getLengthDifference(numDilLayers, initialConvolutionWidths,
                              initialConvolutionWidths[0], verbose)
