from __future__ import print_function
import argparse
import sys  # for commanad line
import os   # for file paths
import re 
import itertools
import cPickle as pickle 
import numpy as np
import rdkit.Chem as Chem 
from rdkit.Chem import Draw
np.random.seed(101) # arbitary, but fixed

def main():
    '''Read reactions'''
    global v

    found_counts = []
    times = []
    candidate_counts = []

    bin_mins = [-100, 5, 50, 200, 500, 1000, 2000, 4000][::-1]
    bin_indices = range(len(bin_mins))
    bin_reactions = [[] for i in bin_indices]

    # Read from file
    print('Reading from file')
    line_no = 0
    while True:
        try:
            (index, reaction_smiles, candidates, time, found_count) = pickle.load(candidates_fid)
            
            for i in bin_indices:
                if found_count >= bin_mins[i]: # this is the right bin
                    bin_reactions[i].append((index, reaction_smiles, found_count, len(candidates)))
                    break

            line_no += 1
            if line_no % 100 == 0:
                print('done {}'.format(line_no))
        except EOFError:
            break
        except KeyboardInterrupt:
            break

    # Report
    N = sum([len(_bin) for _bin in bin_reactions])
    print('Found {} reactions in file'.format(N))

    # Convert to numpy
    N_select = 10
    print('Randomly selecting {} from each'.format(N_select))

    with open(os.path.join(outfolder, 'legend.csv'), 'w') as legend_fid:
        legend_fid.write('Bin index\tIndex inside bin\tIndex\tReaction smiles\tFound count\tNumber of candidates\n')
        for i in bin_indices:
            np.random.shuffle(bin_reactions[i])
            ctr = 0
            for bin_reaction in bin_reactions[i]:
                (index, reaction_smiles, found_count, num_candidates) = bin_reaction
                legend_fid.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    i, ctr, index, reaction_smiles, found_count, num_candidates,
                ))
                reactants = Chem.MolFromSmiles(reaction_smiles.split('>')[0])
                products = Chem.MolFromSmiles(reaction_smiles.split('>')[2])
                if reactants is None or products is None:
                    continue
                [a.ClearProp('molAtomMapNumber') for a in reactants.GetAtoms()]
                [a.ClearProp('molAtomMapNumber') for a in products.GetAtoms()]
                Draw.MolToFile(reactants, os.path.join(outfolder, 'reactants_{}.png'.format(index)), size=(500,500))
                Draw.MolToFile(products, os.path.join(outfolder, 'products_{}.png'.format(index)), size=(500,500))
                ctr += 1
                if ctr == N_select:
                    break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type = str, help = 'File to read candidates from')
    args = parser.parse_args()

    infile = args.infile
    outfolder = os.path.join(os.path.dirname(__file__), 'cases')
    try:
        os.makedir(outfolder)
    except Exception:
        pass

    with open(infile, 'rb') as candidates_fid:
        main()