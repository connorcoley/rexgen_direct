# Import relevant packages
from __future__ import print_function
import argparse
import numpy as np                      # for simple calculations
import os                          # for saving
import sys

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(4)

import rdkit.Chem as Chem
from pymongo import MongoClient    # mongodb plugin
import re
import time
from tqdm import tqdm
import json
import rdkit.Chem.AllChem as AllChem
import cPickle as pickle
from collections import defaultdict

USE_STEREOCHEMISTRY = False 
MINCOUNT = 5

def load():
    '''
    Load templates from file to apply
    '''

    outfile = 'templates.json'
    if os.path.isfile(outfile):
        with open(outfile, 'rb') as fid: 
            template_dict = json.load(fid)
    else:
        raise ValueError('Need a template .json file at {}'.format(outfile))

    templates = []
    for (smarts, count) in template_dict.iteritems():

        if count < MINCOUNT: 
            continue
        
        # Need to convert template back to synth direction
        reaction_smarts_synth = '(' + smarts.split('>')[2] + ')>>(' + smarts.split('>')[0] + ')'
        rxn_f = AllChem.ReactionFromSmarts(str(reaction_smarts_synth))
        
        # Load
        if rxn_f.Validate()[1] == 0:
            templates.append(
                (rxn_f, count)
            )

    return templates

def main(reaction_fid):

    templates = load()
    print('{} templates loaded from mincount {}'.format(len(templates), MINCOUNT))

    new_candidate_file = 'candidates_new.pickle'
    old_candidate_file = 'candidates.pickle'
    done_reactions = 0
    with open(new_candidate_file, 'wb') as candidates_fid:
        if os.path.isfile(old_candidate_file):
            with open(old_candidate_file, 'rb') as old_candidate_fid:
                while True:
                    try:
                        a = pickle.load(old_candidate_fid)
                        pickle.dump(a, candidates_fid)
                        done_reactions += 1
                    except EOFError:
                        break
                    except pickle.UnpicklingError as e:
                        print(e)
                        print('got up to {} done reactions though'.format(done_reactions))
                        break
        print('{} done reactions in file'.format(done_reactions))

        def log_to_file(index = -1, reaction_smiles = '', candidates = [], time = -1, found_count = -1):
            pickle.dump((index, reaction_smiles, candidates, time, found_count), 
                        candidates_fid, pickle.HIGHEST_PROTOCOL)

        # Look for entries
        counter = -1
        for line in reaction_fid:
            counter += 1

            # Don't repeat previously-done reactions
            if counter< done_reactions: 
                continue

            if v: 
                print('##################################')
                print('###        RXN {}'.format(counter))
                print('##################################')


            try:
                # Unpack
                reaction_smiles = str(line.split()[0])
                reactants, products = [Chem.MolFromSmiles(smi) for smi in reaction_smiles.split('>>')]
                Chem.SanitizeMol(reactants)
                Chem.SanitizeMol(products)

            except Exception as e:
                # can't sanitize -> skip
                print('Could not load SMILES or sanitize')
                print(line)
                if p: raw_input('Pause')
                log_to_file(index = counter, 
                            reaction_smiles = reaction_smiles)
                continue

            # Define target w/o atom mapping
            products_nomap = Chem.MolFromSmiles(reaction_smiles.split('>>')[1])
            [a.ClearProp('molAtomMapNumber') for a in products_nomap.GetAtoms()]
            product_target_nomap = Chem.MolToSmiles(products_nomap)

            # Get read to apply templates
            candidates = defaultdict(lambda: -1)
            found_count = -1
            start_time = time.time()

            # Go through templates
            for (rxn_f, count) in templates:
                # Perform transformation
                try:
                    outcomes = rxn_f.RunReactants([reactants])
                except Exception as e:
                    if v: print(e)
                    continue
                if not outcomes: continue # no match
                for j, outcome in enumerate(outcomes):
                    outcome = outcome[0] # all products represented as single mol by transforms
                    try:
                        outcome.UpdatePropertyCache()
                        Chem.SanitizeMol(outcome)
                        [a.SetProp('molAtomMapNumber', a.GetProp('old_molAtomMapNumber')) \
                            for (i, a) in enumerate(outcome.GetAtoms()) \
                            if 'old_molAtomMapNumber' in a.GetPropsAsDict()]
                    except Exception as e:
                        if v: print(e)
                        continue
                    outcome_smiles = Chem.MolToSmiles(outcome)
                    if v: print('Outcome SMILES: {}'.format(outcome_smiles))

                    # Reduce to longest smiles fragment
                    outcome_smiles = max(outcome_smiles.split('.'), key = len)

                    # Save this atom-mapped outcome with the highest count
                    if count > candidates[outcome_smiles]:
                        candidates[outcome_smiles] = count 

                    # Also check map-free to see if we have found the true outcome
                    [a.ClearProp('molAtomMapNumber') for a in outcome.GetAtoms()]
                    outcome_smiles_nomap = Chem.MolToSmiles(outcome)
                    outcome_smiles_nomap = max(outcome_smiles_nomap.split('.'), key = len)
                    if outcome_smiles_nomap == product_target_nomap:
                        if count > found_count:
                            found_count = count 

            # LOGGING
            end_time = time.time()

            print('this was index {}'.format(counter))
            print('this reaction smiles: {}'.format(reaction_smiles))
            print('this generated {} candidates'.format(len(candidates)))
            print('the true one was found with count {}'.format(found_count))
            print('time: {}'.format(end_time - start_time))
            
            if p: raw_input('Pause...')
            log_to_file(index = counter, reaction_smiles = reaction_smiles, 
                        candidates = candidates.items(), time = end_time - start_time, 
                        found_count = found_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type = bool, default = False,
                        help = 'Verbose printing; defaults to False')
    parser.add_argument('-i', '--infile', type = str, help = 'File to read reactions from')
    parser.add_argument('-p', '--pause', type = bool, default = False, 
        help = 'Whether or not to pause, default False')
    args = parser.parse_args()
    v = bool(args.v)
    p = bool(args.pause)

    with open(args.infile) as reaction_fid:
        main(reaction_fid)
