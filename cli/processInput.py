import sys
import argparse
import logging
sys.path.insert(0, "../src")
import input_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_arguments():
    pass 

def run(): 
    # parse arguments
    # run expansion and filterings for overlaps
    # run filter for Ns
    # divide the set in train, validation and testing sets
    # get one hot encoded sequences from each set
    # save everything in H5 training, validation and testing files. 
    pass

if __name__=="__main__": 
    pass


