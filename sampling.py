#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 12:49:46 2018

@author: ififsun
"""

import numpy as np


def get_indice_graph(adj, mask, size, keep_r=1.0):
    indices = mask.nonzero()[0]
    #droupout
    if keep_r < 1.0:
        indices = np.random.choice(indices, int(indices.size*keep_r), False)
    pre_indices = set()
    indices = set(indices)
    while len(indices) < size:
        new_add = indices - pre_indices
        if not new_add:
            break
        pre_indices = indices
        candidates = get_candidates(adj, new_add) - indices
        if len(candidates) > size - len(indices):
            candidates = set(np.random.choice(list(candidates), size-len(indices), False))
        indices.update(candidates)
    print('indices size:-------------->', len(indices))
    return sorted(indices)


def get_sampled_index(adj, size, center_num=1):
    n = adj.shape[0]
    pre_indices = set()
    indices = set(np.random.choice(n, center_num, False))
    while len(indices) < size:
        if len(pre_indices) != len(indices):
            new_add = indices - pre_indices
            pre_indices = indices
            candidates = get_candidates(adj, new_add) - indices
        else:
            candidates = random_num(n, center_num, indices)
        sample_size = min(len(candidates), size-len(indices))
        if not sample_size:
            break
        if len(candidates) > size - len(indices):
            candidates = set(np.random.choice(list(candidates), size-len(indices), False))
        indices.update(candidates)
    return sorted(indices)


def get_candidates(adj, new_add):
    # select the point that adjancy to those point
    return set(adj[sorted(new_add)].sum(axis=0).nonzero()[1])


def random_num(n, num, indices):
    # sample from #num data that not in indices
    cans = set(np.arange(n)) - indices
    num = min(num, len(cans))
    if len(cans) == 0:
        return set()
    new_add = set(np.random.choice(list(cans), num, replace=False))
    return new_add
