import abacus_cosmos.Halos as ach
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import glob
from collections import Counter
import scipy.ndimage as scpimg

import glob


def find_watershed(divergence_filename):
    print("processing: {}".format(divergence_filename))
    divergence_field = np.load(divergence_filename)

    sort_index = np.argsort(divergence_field.flatten(order='C'))
    watershed_group = np.ones(np.shape(divergence_field), dtype=int)*-1
    N_side = np.shape(divergence_field)[0]
    print("N_side={}".format(N_side))
    n_total = len(sort_index)

    n_group = 0
    n_cell = 0
    for l in range(len(sort_index)):
        n_cell += 1
        ijk = np.unravel_index(sort_index[l], np.shape(divergence_field), order='C')
        groups = []
        for i in [-1,0, 1]:
            for j in [-1,0, 1]:
                for k in [-1,0, 1]:
                    ijk_test = tuple((np.array(ijk)+(i, j, k))%N_side)
                    groups.append(watershed_group[ijk_test])
        c = Counter(groups)
    
        if -1 in c.keys():
            c.pop(-1)
        if len(c)==0:
            watershed_group[ijk] = n_group
            n_group += 1
        else:
            g = c.most_common(1)
            watershed_group[ijk] = g[0][0]
        
        if((n_cell%100000)==0):
            print("{:.2f} % of total. {} groups".format(100*n_cell/n_total,n_group))
    filename = divergence_filename.replace("div_", "watershed_")
    np.save(filename, watershed_group)
    return 
    
compute_all = True

if compute_all:
    div_files = glob.glob("../data/AbacusCosmos_720box_planck_00_0_rockstar_halos/z0.1/fields/div_*.npy")
    print(div_files)
    for div_file in div_files:
        find_watershed(div_file)
