from __future__ import print_function
import argparse
import sys  # for commanad line
import os   # for file paths
import re 
import itertools
import cPickle as pickle 
import numpy as np

def main():
    '''Read reactions'''
    global v

    found_counts = []
    times = []
    candidate_counts = []

    f = open(infile + '_summary.csv', 'w')

    # Read from file
    print('Reading from file')
    line_no = 0
    while True:
        try:
            (index, reaction_smiles, candidates, time, found_count) = pickle.load(candidates_fid)
            
            f.write('{}\t{}\t{}\t{}\n'.format(index, reaction_smiles, found_count, len(candidates)))

            line_no += 1
            if line_no % 100 == 0:
                print('done {}'.format(line_no))
        except EOFError:
            break
        except KeyboardInterrupt:
            break
        finally:
            f.close()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type = str, help = 'File to read candidates from')
    args = parser.parse_args()

    infile = args.infile

    with open(infile, 'rb') as candidates_fid:
        main()