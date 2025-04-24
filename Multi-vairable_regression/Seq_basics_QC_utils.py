import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

from dnachisel import *
from dnachisel import biotools

#### Global ####
nat_aa = 'ARNDCQEGHILKMFPSTWYV'
flex_pool = 'GSAT'
pos_charged_aa = 'RHK'
neg_charged_aa = 'DE'
aliphatic_aa = 'VLIA'

#### Utilities ####
def hinge_rand_generator(length, seed=None):
    """ Generate random protein sequence of specific length on 20 aas.

    Args:
        length (int): len of protein
        seed (, optional): Defaults to None.

    Returns:
        str: returned seq
    """
    
    rand_pr = random_protein_sequence(length+2, seed=None)   # +2 is because the start and stop codon
    return rand_pr[1:-1]

def hinge_len_ctrl(pr, trgt_len, place_holder='X'):
    """Contrl the input sequence to a certain length.
    if trgt_len < len(pr): print notice and truncate
    if trgt_len >= len(pr): fill with 'X'

    Args:
        pr (str): protein sequence
        trgt_len (int): 
        place_holder (str, optional): Defaults to 'X'. Place holder to make seq longer

    Returns:
        str:
    """
    if trgt_len < len(pr):
        print('Sequence shorter than expected. Turncated')
        res = pr[:trgt_len]
    else:
        res = pr + place_holder*(trgt_len-len(pr))
    
    return res

def point_mutation(pr, mut_index, sub_pool=nat_aa):
    """Make a point mutation at a certain location. aa after mutation should from sub_pool

    Args:
        pr (str): pr to mutate
        index (int): which index aa to mutate
        sub_pool (str, optional): Defaults to 'ARNDCQEGHILKMFPSTWYV' (which is global var. nat_aa).
    """
    if mut_index >= len(pr):
        raise Exception('Invalid index for mutation.')
    else:
        pr = list(pr)
        
        if pr[mut_index] in sub_pool:
            sub_pool = sub_pool.replace(pr[mut_index], '')  # update subpool if the aa to mutate is inside
        
        if len(sub_pool) > 0:
            pr[mut_index] = sub_pool[np.random.randint(0, len(sub_pool))]
    
    return ''.join(pr)

def percentage_mutation(pr, percentage, sub_pool=nat_aa):
    """Mutate x percent of the sequence with a sub_pool of aa

    Args:
        pr (str): pr to mutate
        sub_pool (str, optional): Defaults to 'ARNDCQEGHILKMFPSTWYV' (which is global var. nat_aa).
    """
    num_of_muts = int(percentage*len(pr))
    mut_idx = random.choices(np.arange(len(pr)), k=num_of_muts)
    
    for i in mut_idx:
        pr = point_mutation(pr, i, sub_pool=sub_pool)
    
    return pr

##### Flexible hinge generation #####
def flexible_rand_generator(length, weight=[25, 25, 25, 25], flex_pool=flex_pool, seed=None):
    """Used to generate a random flexible linker from a pool of small residues. By default is 'GSAT'

    Args:
        length (int): length of random sequence
        seed (seed): for np.random
        weight (list): weight of each residue in pool. Must be same length as pool. Must be summed as 100
        flex_pool (str, optional): Defaults to 'GSAT'.

    Returns:
        str: final sequence
    """
    
    # check if different seed needed
    if seed is not None:
        np.random.seed(seed)

    # check if weight and flex_pool has the same length
    if len(flex_pool) != len(weight):
        raise Exception('Input weight and flex_pool have different length.')
    
    # check if weight can be summed as 100
    if np.sum(weight) != 100:
        raise Exception('Weight sum is not 100%.')
    else:
        flex_pool_dict = dict(zip(list(flex_pool), weight))

    flex_pool_weighted = ''.join([x*flex_pool_dict[x] for x in flex_pool])

    seq = np.random.choice(list(flex_pool_weighted), length)
    
    return ''.join(seq)

def motif_repeat(motif, num_repeat):
    """Used to repeat a motif for many times

    Args:
        motif (str): 
        num_repeat (int):
    """
    return motif*num_repeat
##### Flexible hinge generation #####

##### X-Pro hinge generation #####
def xp_rand_generator(repeat, x_pool='AS', weight=[50, 50], seed=None):
    """Used to generate n*XPs rigid linkers.

    Args:
        repeat (int): number of repeats
        x_pool (str, optional): Defaults to 'AS'.
        weight (list, optional): Defaults to [50, 50].
        seed: Defaults to None
    """
    # check if different seed needed
    if seed is not None:
        np.random.seed(seed)

    # check if weight and flex_pool has the same length
    if len(x_pool) != len(weight):
        raise Exception('Input weight and flex_pool have different length.')
    
    # check if weight can be summed as 100
    if np.sum(weight) != 100:
        raise Exception('Weight sum is not 100%.')
    else:
        x_pool_dict = dict(zip(list(x_pool), weight))
        
    x_pool_weighted = ''.join([a*x_pool_dict[a] for a in x_pool])
    
    seq = []
    for i in np.arange(repeat):
        seq.append(np.random.choice(list(x_pool_weighted))+'P')
        
    return ''.join(seq)
##### X-Pro hinge generation #####

##### Helical hinge generation #####
def pool_rand_generator(length, pool=aliphatic_aa, weight=[25, 25, 25, 25], seed=None):
    """ Used to generate a randomized sequence from a weighted amino acid pool

    Args:
        len (int): number of randomnized sequence
        pool (str, optional): Defaults to aliphatic_aa = 'VLIA'.
        weight (list, optional): Defaults to [25, 25, 25, 25].
        seed (None, optional): Defaults to None.
    """
    # check if different seed needed
    if seed is not None:
        np.random.seed(seed)

    # check if weight and flex_pool has the same length
    if len(pool) != len(weight):
        raise Exception('Input weight and flex_pool have different length.')
    
    # check if weight can be summed as 100
    if np.sum(weight) != 100:
        raise Exception('Weight sum is not 100%.')
    else:
        pool_dict = dict(zip(list(pool), weight))
    
    pool_weighted = ''.join([a*pool_dict[a] for a in pool])

    return ''.join(np.random.choice(list(pool_weighted), length))

def helical_motif_rand_generator(neg_pool=neg_charged_aa, neg_weight=[50,50], aliphatic_pool=aliphatic_aa, 
                                 aliphatic_weight=[25,25,25,25], pos_pool=pos_charged_aa,
                                 pos_weight=[50,0,50], seed=None):
    """Used to generate a helical "EAAAK" like motif with 1 neg aa, 3 aliphatic aas, 1 pos aa.

    Args:
        neg_pool (_type_, optional): _description_. Defaults to neg_charged_aa.
        neg_weight (list, optional): _description_. Defaults to [50,50].
        aliphatic_pool (_type_, optional): _description_. Defaults to aliphatic_aa.
        aliphatic_weight (list, optional): _description_. Defaults to [25,25,25,25].
        pos_pool (_type_, optional): _description_. Defaults to pos_charged_aa.
        pos_weight (list, optional): _description_. Defaults to [50,0,50].

    Returns:
        _type_: _description_
    """
    a = pool_rand_generator(1, pool=neg_pool, weight=neg_weight, seed=seed)
    b = pool_rand_generator(3, pool=aliphatic_pool, weight=aliphatic_weight, seed=seed)
    c = pool_rand_generator(1, pool=pos_pool, weight=pos_weight, seed=seed)
    
    return ''.join([a,b,c])

def helical_motif_repeat(num_repeat):
    """Used to stitch randomized helical motifs from helical_motif_rand_generator

    Args:
        num_repeat (int):
    """
    seq = []
    for i in np.arange(num_repeat):
        seq.append(helical_motif_rand_generator())
    
    return ''.join(seq)
##### Helical hinge generation #####

##### AA frequency calculation #####
# 2 functions
def amino_acid_count(x):
    """get amino acid counts for a given sequence string
    """
    freq = {}
    for c in set(x):
       freq[c] = x.count(c)
    return freq

def get_seq_helix_panelty(pr, helix_ref):
    """get a total panelty value for a given sequence string
    """
    seq = amino_acid_count(pr)
    res = 0
    for aa in list(seq.keys()):
        tmp = float(
            helix_ref[helix_ref['1-letter']==aa]['helical_panelty(kj/mol)'])
        res += seq[aa]*tmp
    return res

def get_aa_freq(aa, x):
    """get the amino acid percentage for a certain aa in a sequence string
    """
    length = len(x)    # get the total length of the string
    freq = amino_acid_count(x)  # get a dict of aa in this string
    
    if aa in list(freq.keys()):
        return freq[aa]/length
    else:
        return 0/length