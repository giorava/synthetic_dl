#!/bin/bash 

# this numbers should be determined from the choosen arquitecture!
OUTPUT_SIZE=2000
INPUT_SIZE=2000

PEAKS_PATH="$HOME/scratch/synthetic_dl/demos/example_datasets/peaks_merged.bed"
CHROM_SIZES="$HOME/reference_genomes/mm10.chrom.sizes"
GENOME="$HOME/reference_genomes/mm10.fa"
TRAIN_CHROM="chr5,chr6,chr7,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19"
TEST_CHROM="chr1,chr8,chr9"
VAL_CHROM="chr2,chr3,chr4"
POS_BIGWIG="$HOME/scratch/synthetic_dl/demos/example_datasets/pos.strand.bigwigs"
NEG_BIGWIG="$HOME/scratch/synthetic_dl/demos/example_datasets/neg.strand.bigwigs"
OUTPUT_PATH="$HOME/scratch/synthetic_dl/demos/data_dl"

mkdir -p $OUTPUT_PATH


python cli/processInput.py \
    --peaks ${PEAKS_PATH} \
    --chrom_sizes ${CHROM_SIZES} \
    --genome_fasta ${GENOME} \
    --output_size ${OUTPUT_SIZE} \
    --pos_bigwig ${POS_BIGWIG} \
    --neg_bigwig ${NEG_BIGWIG} \
    --trainset ${TRAIN_CHROM} \
    --testset ${TEST_CHROM} \
    --valset ${VAL_CHROM} \
    --output_path ${OUTPUT_PATH} \
    --input_size ${INPUT_SIZE}