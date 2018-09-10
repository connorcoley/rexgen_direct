import tensorflow as tf
from nn import linearND, linear
from mol_graph_direct_useScores import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph, smiles2graph, bond_types
from models import *
import math, sys, random
from optparse import OptionParser
import threading
from multiprocessing import Queue
import rdkit
from rdkit import Chem
import os
import numpy as np 

TOPK = 100

hidden_size = 500
depth = 3
core_size = 16
MAX_NCAND = 1500
model_path = os.path.join(os.path.dirname(__file__), "model-core16-500-3-max150-direct-useScores", "model.ckpt-2400000")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class DirectCandRanker():
    def __init__(self):
        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        _input_atom = tf.placeholder(tf.float32, [None, None, adim])
        _input_bond = tf.placeholder(tf.float32, [None, None, bdim])
        _atom_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
        _bond_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
        _num_nbs = tf.placeholder(tf.int32, [None, None])
        _core_bias = tf.placeholder(tf.float32, [None])
        self._src_holder = [_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs, _core_bias]

        q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32])
        self.enqueue = q.enqueue(self._src_holder)
        input_atom, input_bond, atom_graph, bond_graph, num_nbs, core_bias = q.dequeue()

        input_atom.set_shape([None, None, adim])
        input_bond.set_shape([None, None, bdim])
        atom_graph.set_shape([None, None, max_nb, 2])
        bond_graph.set_shape([None, None, max_nb, 2])
        num_nbs.set_shape([None, None])
        core_bias.set_shape([None])

        graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs) 
        with tf.variable_scope("mol_encoder"):
            fp_all_atoms = rcnn_wl_only(graph_inputs, hidden_size=hidden_size, depth=depth)

        reactant = fp_all_atoms[0:1,:]
        candidates = fp_all_atoms[1:,:]
        candidates = candidates - reactant
        candidates = tf.concat([reactant, candidates], 0)

        with tf.variable_scope("diff_encoder"):
            reaction_fp = wl_diff_net(graph_inputs, candidates, hidden_size=hidden_size, depth=1)

        reaction_fp = reaction_fp[1:]
        reaction_fp = tf.nn.relu(linear(reaction_fp, hidden_size, "rex_hidden"))

        score = tf.squeeze(linear(reaction_fp, 1, "score"), [1]) + core_bias # add in bias from CoreFinder

        tk = tf.minimum(TOPK, tf.shape(score)[0])
        pred_topk_scores, pred_topk = tf.nn.top_k(score, tk)

        self.predict_vars = [pred_topk_scores, pred_topk]

        tf.global_variables_initializer().run(session=self.session)


    def restore(self, model_path=model_path):
        saver = tf.train.Saver()
        saver.restore(self.session, model_path)

    
    def predict(self, react, top_cand_bonds, top_cand_scores=[]):
        '''react: atom mapped reactant smiles
        top_cand_bonds: list of strings "ai-aj-bo"'''

        cand_bonds = []
        if not top_cand_scores:
            top_cand_scores = [0.0 for b in top_cand_bonds]
        for i, b in enumerate(top_cand_bonds):
            x,y,t = b.split('-')
            x,y,t = int(float(x))-1,int(float(y))-1,float(t)

            cand_bonds.append((x,y,t,float(top_cand_scores[i])))

        while True:
            src_tuple,conf = smiles2graph(react, None, cand_bonds, None, core_size=core_size, cutoff=MAX_NCAND, testing=True)
            if len(conf) <= MAX_NCAND:
                break
            ncore -= 1

        feed_map = {x:y for x,y in zip(self._src_holder, src_tuple)}
        self.session.run(self.enqueue, feed_dict=feed_map)

        cur_pred_scores, cur_pred = self.session.run(self.predict_vars)
        

        idxfunc = lambda a: a.GetAtomMapNum()
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                      Chem.rdchem.BondType.AROMATIC]
        bond_types_as_double = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}

        # Don't waste predictions on bond changes that aren't actually changes
        rmol = Chem.MolFromSmiles(react)
        rbonds = {}
        for bond in rmol.GetBonds():
            a1 = idxfunc(bond.GetBeginAtom())
            a2 = idxfunc(bond.GetEndAtom())
            t = bond_types.index(bond.GetBondType()) + 1
            a1,a2 = min(a1,a2),max(a1,a2)
            rbonds[(a1,a2)] = t

        outcomes = []
        for i in range(len(cur_pred)):
            idx = cur_pred[i]
            cbonds = []
            # Define edits from prediction
            for x,y,t,v in conf[idx]:
                x,y = x+1,y+1
                if ((x,y) not in rbonds and t > 0) or ((x,y) in rbonds and rbonds[(x,y)] != t):
                    cbonds.append((x, y, bond_types_as_double[t]))
            pred_smiles = edit_mol(rmol, cbonds)
            outcomes.append((cur_pred_scores[i], '.'.join(set(pred_smiles))))

        all_scores = softmax(np.array([x[0] for x in outcomes]))

        for i in range(len(outcomes)):
            outcomes[i] = (all_scores[i], outcomes[i][1])

        return outcomes

    def predict_smiles(self, *args, **kwargs):
        '''Wrapper to canonicalize and get SMILES probs'''
        outcomes = self.predict(*args, **kwargs)
        this_reactants_smiles = args[0]

        from collections import defaultdict
        outcomes_to_ret = defaultdict(float)
        reactants_smiles_split = this_reactants_smiles.split('.')
        for sco, outcome in outcomes:
            smiles_list = set(outcome.split('.'))
            
            # Canonicalize
            smiles_canonical = set()
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    continue
                smiles_canonical.add(Chem.MolToSmiles(mol))

            # Remove unreacted frags
            smiles_canonical = smiles_canonical - set(reactants_smiles_split)
            if not smiles_canonical:
                continue # no reaction?

            smiles = max(smiles_canonical, key=len) # NOTE: this is not great...byproducts may be longer
            outcomes_to_ret[smiles] += sco

        # Renormalize and re-rank
        outcomes = sorted(outcomes_to_ret.items(), key=lambda x: x[1])
        total_prob = sum([outcome[1] for outcome in outcomes])
        for i, outcome in enumerate(outcomes):
            outcome = (outcome[0], outcome[1] / total_prob)
            outcomes[i] = outcome

        return outcomes

if __name__ == '__main__':

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    print(sys.path)

    from rexgen_direct.core_wln_global.directcorefinder import DirectCoreFinder 
    from rexgen_direct.scripts.eval_by_smiles import edit_mol

    directcorefinder = DirectCoreFinder()
    directcorefinder.restore()
    if len(sys.argv) < 2:
        print('Using example reaction')
        react = '[CH3:26][c:27]1[cH:28][cH:29][cH:30][cH:31][cH:32]1.[Cl:18][C:19](=[O:20])[O:21][C:22]([Cl:23])([Cl:24])[Cl:25].[NH2:1][c:2]1[cH:3][cH:4][c:5]([Br:17])[c:6]2[c:10]1[O:9][C:8]([CH3:11])([C:12](=[O:13])[O:14][CH2:15][CH3:16])[CH2:7]2'
        print(react)
    else:
        react = str(sys.argv[1])

    (react, bond_preds, bond_scores, cur_att_score) = directcorefinder.predict(react)

    directcandranker = DirectCandRanker()
    directcandranker.restore()
    #outcomes = directcandranker.predict(react, bond_preds, bond_scores)
    outcomes = directcandranker.predict_smiles(react, bond_preds, bond_scores)
    for outcome in outcomes:
        print(outcome)


