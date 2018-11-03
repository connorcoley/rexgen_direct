from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole, ReactionToImage, MolToImage, MolsToGridImage
import rdkit.Chem.AllChem as AllChem
from IPython.display import SVG, display, clear_output
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
IPythonConsole.ipython_useSVG = True
IPythonConsole.molSize = (800, 400)
import pandas as pd
import os 

rexgen_direct_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'rexgen_direct')

cols = ['reactants', 'products', 'product_smiles', 'product_smiles_sani', 'rank (10 if not found)'] + ['pred{}'.format(i) for i in range(10)] + ['rank (10 if not found) sani'] + ['pred{} sani'.format(i+1) for i in range(10)]
df_rankpred = pd.read_csv(os.path.join(rexgen_direct_root, 'rank_diff_wln/model-core16-500-3-max150-direct-useScores/test.cbond_detailed_2400000.eval_by_smiles'), sep="\t",header=None, names=cols) 

import tensorflow as tf
print(tf.__version__)
import math, sys, random
from collections import Counter
from optparse import OptionParser
from functools import partial
import threading
from multiprocessing import Queue
import os
from collections import defaultdict
from rdkit import Geometry

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole, ReactionToImage, MolToImage, MolsToGridImage
import rdkit.Chem.AllChem as AllChem
from PIL import Image, ImageOps
import rdkit.Chem.Draw as Draw

from IPython.display import SVG, display, clear_output
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

import sys
print(sys.path)
sys.path += [os.path.dirname(rexgen_direct_root)] # need to be able to import directcorefinder

from rexgen_direct.core_wln_global.directcorefinder import DirectCoreFinder
directcorefinder = DirectCoreFinder()
directcorefinder.load_model()

def arrow():
    subImgSize = (115, 115)
    img, canvas = Draw._createCanvas(subImgSize)
    p0 = (10, subImgSize[1]//2)
    p1 = (subImgSize[0]-10, subImgSize[1]//2)
    p3 = (subImgSize[0]-20, subImgSize[1]//2-10)
    p4 = (subImgSize[0]-20, subImgSize[1]//2+10)
    
    
    def drawthick(pa, pb, d=1):
        canvas.addCanvasLine(pa, pb, color=(0, 0, 0))
        pa = (pa[0], pa[1]+d)
        pb = (pb[0], pb[1]+d)
        canvas.addCanvasLine(pa, pb, color=(0, 0, 0))
        
    drawthick(p0, p1)
    drawthick(p3, p1, d=-1)
    drawthick(p4, p1)
    
    if hasattr(canvas, 'flush'):
        canvas.flush()
    else:
        canvas.save()
    return img

def TrimImgByWhite(img, padding=0):
    '''This function takes a PIL image, img, and crops it to the minimum rectangle 
    based on its whiteness/transparency. 5 pixel padding used automatically.'''

    # Convert to array
    as_array = np.array(img)  # N x N x (r,g,b,a)

    # Content defined as non-white and non-transparent pixel
    has_content = np.sum(as_array, axis=2, dtype=np.uint32) != 255 * 3
    xs, ys = np.nonzero(has_content)

    # Crop down
    margin = 10
    x_range = max([min(xs) - margin, 0]), min([max(xs) + margin, as_array.shape[0]])
    y_range = max([min(ys) - margin, 0]), min([max(ys) + margin, as_array.shape[1]])
    as_array_cropped = as_array[
        x_range[0]:x_range[1], y_range[0]:y_range[1], 0:3]

    img = Image.fromarray(as_array_cropped, mode='RGB')

    return ImageOps.expand(img, border=padding, fill=(255, 255, 255))

def StitchPILsHorizontally(imgs):
    '''This function takes a list of PIL images and concatenates
    them onto a new image horizontally, with each one
    vertically centered.'''

    # Create blank image (def: transparent white)
    heights = [img.size[1] for img in imgs]
    height = max(heights)
    widths = [img.size[0] for img in imgs]
    width = sum(widths)
    res = Image.new('RGBA', (width, height), (255, 255, 255, 255))

    # Add in sub-images
    for i, img in enumerate(imgs):
        offset_x = sum(widths[:i])  # left to right
        offset_y = (height - heights[i]) // 2
        res.paste(img, (offset_x, offset_y))

    return res


fontsize = 0.6
linewidth = 5

def analyze_reaction(react, product=None, guessed_prod=None, clearmap=False, atts=[], max_bond_preds=5, max_att_score_info=10, showmap=True, showhighlight=True):
    if clearmap:
        m = Chem.MolFromSmiles(react)
        [a.ClearProp('molAtomMapNumber') for a in m.GetAtoms()]
        [a.SetIntProp('molAtomMapNumber', i+1) for i, a in enumerate(m.GetAtoms())]
        react = Chem.MolToSmiles(m)
    (react, bond_preds, bond_scores, cur_att_score) = directcorefinder.predict(react)
    react_mol = Chem.MolFromSmiles(react)
    map_to_idx = {}; idx_to_map = {}
    for i, a in enumerate(react_mol.GetAtoms()):
        map_to_idx[a.GetIntProp('molAtomMapNumber')] = i
        idx_to_map[i] = a.GetIntProp('molAtomMapNumber')
        
    for i in range(min(len(bond_preds), max_bond_preds)):
         print('Prediction {}:    {:15} with score {:6}'.format(i+1, bond_preds[i], bond_scores[i]))
    
    highlight = []
    scores = {}
    for a in atts:
        att_atom_mapnum = a
        print('Atom ID {} attends to...'.format(a))
        a = a - 1
        attention_scores = cur_att_score[a,:, 0]
        for j in np.argsort(attention_scores)[::-1][:max_att_score_info]:
            print('    atom ID {:3} with score {:6}'.format(j+1, attention_scores[j]))
            
        for j, sco in enumerate(attention_scores):
            highlight.append(j+1)
            scores[j+1] = float(sco)
        
    colors = {}
    if atts:
        scale = 0.8
        for k in scores:
            colors[k] = (1-scale*scores[k], 1-scale*scores[k], 1)
        colors[att_atom_mapnum] = (0, 1, 0)

        
    # Iterate through each reactant
    imgs = []
    for smi_frag in react.split('.'):
        mol = Chem.MolFromSmiles(smi_frag)
        try:
            Chem.Kekulize(mol)
        except:
            pass
        map_to_idx = {}; idx_to_map = {}
        for i, a in enumerate(mol.GetAtoms()):
            map_to_idx[a.GetIntProp('molAtomMapNumber')] = i
            idx_to_map[i] = a.GetIntProp('molAtomMapNumber')
        
        this_highlight = [map_to_idx[mapnum] for mapnum in highlight if mapnum in map_to_idx]
        this_colors = {map_to_idx[mapnum]: colors[mapnum] for mapnum in colors if mapnum in map_to_idx}
        
        def get_scaled_drawer(mol):
            dpa = 26
            rdDepictor.Compute2DCoords(mol)
            conf = mol.GetConformer()
            xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
            ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
            
            point_min = Geometry.rdGeometry.Point2D()
            point_max = Geometry.rdGeometry.Point2D()
            point_min.x = min(xs) - 3
            point_min.y = min(ys) - 3
            point_max.x = max(xs) + 3
            point_max.y = max(ys) + 3
            
            drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
            w = int(dpa * (point_max.x - point_min.x))
            h = int(dpa * (point_max.y - point_min.y))
            
            drawer.SetScale(w, h, point_min, point_max)
            return drawer


        drawer = get_scaled_drawer(mol)
        drawer.SetFontSize(14)
        opts = drawer.drawOptions()
        opts.useBWAtomPalette()
        opts.additionalAtomLabelPadding = 0.15
        
    
        
        for i, a in enumerate(mol.GetAtoms()):
            symbol = a.GetSymbol()
            if a.GetFormalCharge():
                symbol += '-'*(-a.GetFormalCharge()) if a.GetFormalCharge() < 0 else '+'*a.GetFormalCharge()
            if symbol == 'C' and not showmap:
                symbol = ''
            if showmap:
                opts.atomLabels[i] = '{}{}'.format(symbol, a.GetProp('molAtomMapNumber'))
            a.ClearProp('molAtomMapNumber')
        
        
        DrawingOptions.bondLineWidth=linewidth
        drawer.SetFontSize(fontsize)
        drawer.DrawMolecule(mol,highlightAtoms=this_highlight,highlightBonds=[],highlightAtomColors=this_colors)
        drawer.FinishDrawing()
        pngtext = drawer.GetDrawingText()
        with open('temp.png', 'wb') as fid:
            fid.write(pngtext)
        img = Image.open('temp.png')
        np.array(img) # magic fix for drawing bug in Python 3.5.3 (?) no clue why this is necessary, but it is
        imgs.append(img)
        
    if product:
        # Add forward reaction arrow
        imgs.append(arrow())

        # Add product
        mol = Chem.MolFromSmiles(product)
        
        try:
            Chem.Kekulize(mol)
        except:
            pass
        if not clearmap:
            map_to_idx = {};
            for i, a in enumerate(mol.GetAtoms()):
                map_to_idx[a.GetIntProp('molAtomMapNumber')] = i
            if atts and att_atom_mapnum in map_to_idx:
                this_highlight = [map_to_idx[att_atom_mapnum]]
                this_colors = {this_highlight[0]: (0, 1, 0)}
            else:
                this_highlight = []
                this_colors = []
        else:
            this_highlight = []; this_colors = []
        
        drawer = get_scaled_drawer(mol)
        opts = drawer.drawOptions()

        for i, a in enumerate(mol.GetAtoms()):
            symbol = a.GetSymbol()
            if a.GetFormalCharge():
                symbol += '-'*(-a.GetFormalCharge()) if a.GetFormalCharge() < 0 else '+'*a.GetFormalCharge()
            if symbol == 'C' and not showmap:
                symbol = ''
            if showmap:
                opts.atomLabels[i] = '{}{}'.format(symbol, a.GetProp('molAtomMapNumber'))
            a.ClearProp('molAtomMapNumber')
            
        opts.useBWAtomPalette()
        DrawingOptions.bondLineWidth=linewidth
        drawer.SetFontSize(fontsize)
        highlight=[]; highlightBonds=[]; colors=[]
        
        
        drawer.DrawMolecule(mol,highlightAtoms=this_highlight,highlightBonds=[],highlightAtomColors=this_colors)
        drawer.FinishDrawing()
        pngtext = drawer.GetDrawingText()
        with open('test.png', 'wb') as fid:
            fid.write(pngtext)
        img = Image.open('test.png')
        np.array(img) # magic fix for drawing bug in Python 3.5.3 (?) no clue why this is necessary, but it is
        imgs.append(img)
    
        
    if guessed_prod:
        mol = Chem.MolFromSmiles(guessed_prod)
        try:
            Chem.Kekulize(mol)
        except:
            pass
        drawer = get_scaled_drawer(mol)
        opts = drawer.drawOptions()
        opts.useBWAtomPalette()
        DrawingOptions.bondLineWidth=linewidth
        drawer.SetFontSize(fontsize)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        pngtext = drawer.GetDrawingText()
        with open('test.png', 'wb') as fid:
            fid.write(pngtext)
        img = Image.open('test.png')
        newimg = []
        for color in img.getdata():
            newimg.append((255, color[1], color[2]))
        newimg_img = Image.new(img.mode, img.size)
        newimg_img.putdata(newimg)
        np.array(newimg_img) # magic fix for drawing bug in Python 3.5.3 (?) no clue why this is necessary, but it is
        imgs.append(newimg_img)
        
        
        
    overall_reaction = StitchPILsHorizontally([TrimImgByWhite(img) for img in imgs])
    display(overall_reaction)
    return overall_reaction

reactants = '[CH2:29]([Sn:30]([CH2:31][CH2:32][CH2:33][CH3:39])([c:34]1[o:35][cH:36][cH:37][n:38]1)[CH2:40][CH2:41][CH2:42][CH3:43])[CH2:44][CH2:45][CH3:46].[CH2:47]1[O:48][CH2:49][CH2:50][O:51][CH2:52]1.[Cu:130][I:131].[F:1][c:2]1[c:3]([O:21][S:22]([C:23]([F:24])([F:25])[F:26])(=[O:27])=[O:28])[cH:4][c:5](-[c:8]2[cH:9][n:10][c:11]3[n:12]2[n:13][cH:14][c:15]([C:17]([F:18])([F:19])[F:20])[n:16]3)[cH:6][cH:7]1'
product = 'CCC'
guessed_prod= 'CC'
reaction = analyze_reaction(reactants, product=product, guessed_prod=guessed_prod, clearmap=True, showmap=False, atts=[3,])


def do_index(index, max_bond_preds=10, df_rankpred=df_rankpred, showdets=True, **kwargs):
    entry = df_rankpred.loc[df_rankpred.index == index]
    print('THIS IS TEST EXAMPLE {} (1-indexed)'.format(entry.index.item()+1))
    print(list(entry['rank (10 if not found)'])[0])
    if showdets:
        print(list(entry['pred0'])[0])
        print(list(entry['pred1'])[0])
        print(list(entry['pred2'])[0])
        print(list(entry['pred3'])[0])
        print(list(entry['pred4'])[0])
        print('{}>>{}'.format(list(entry['reactants'])[0], list(entry['products'])[0]))
    save = kwargs.pop('save', None)
    
    if list(entry['rank (10 if not found)'])[0] != 1:
        guessed_prods = list(entry['pred0'])[0].split('.')
        rct_mol = Chem.MolFromSmiles(list(entry['reactants'])[0])
        [a.ClearProp('molAtomMapNumber') for a in rct_mol.GetAtoms()]
        reactants = Chem.MolToSmiles(rct_mol)
        prod_list = [prod for prod in guessed_prods if prod not in reactants]
        if prod_list:
            guessed_prod = max(prod_list, key=len)
        else:
            guessed_prod = '[No]'
    else:
        guessed_prod = None
        
    img = analyze_reaction(list(entry['reactants'])[0], product=list(entry['products'])[0], guessed_prod=guessed_prod, max_bond_preds=max_bond_preds, **kwargs)
    
    if save:
        img.save(save + '.png', format='png', quality=100)
        
    return img