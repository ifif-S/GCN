import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import random
from collections import defaultdict as dd
from scipy.sparse import csgraph

def parse_index_file(filename):
    index = []

    for line in open(filename):
        index.append(int(line.strip()))

    return index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

def laplacian(mx, norm):
    """Laplacian-normalize sparse matrix"""
    assert (all (len(row) == len(mx) for row in mx)), "Input should be a square matrix"

    return csgraph.laplacian(adj, normed = norm)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_data(path="./data", dataset="cora"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)

    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)

    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    print("\n[STEP 1]: Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding="latin1"))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        #Citeseer dataset contains some isolated nodes in the graph
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum()/2))

    features = normalize(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)
    adj = torch.FloatTensor(np.array(adj.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end+1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)

    return adj, features, labels, idx_train, idx_val, idx_test, allx, y, x.shape[0],graph

def gen_label_graph(x_size,nclass, allx, y):
    """generator for batches for label context loss.
    """
    labels, label2inst, not_label = [], dd(list), dd(list)
    for i in range(x_size):
        flag = False
        for j in range(nclass):
            if y[i, j] == 1 and not flag:
                labels.append(j)
                label2inst[j].append(i)
                flag = True
            elif y[i, j] == 0:
                not_label[j].append(i)

    while True:
        g = []
        ind = np.random.permutation(x_size)
        for i in range(len(ind)):        
            x1 = ind[i]
            label = labels[x1]
            if len(label2inst) == 1: continue
            x2 = random.choice(label2inst[label])
            g.append([x1, x2])
        g = np.array(g, dtype = np.int32)
        yield allx[g[:, 0]], g[:, 1]
        
def gen_graph( graph, features, path_size = 10, window_size=3):
    """generator for batches for graph context loss.
    """
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    num_x = adj.shape[0]
    while True:
        ind = np.random.permutation(num_x)
        g = []
        for k in range(len(ind)):
            if len(graph[k]) == 0: continue
            path = [k]
            for _ in range(path_size):
                path.append(random.choice(graph[path[-1]]))
                for l in range(len(path)):
                    if path[l] >= num_x: continue
                    for m in range(l - window_size, l + window_size + 1):
                        if m < 0 or m >= len(path): continue
                        if path[m] >= num_x: continue
                        g.append([path[l], path[m]])
        g = np.array(g, dtype = np.int32)
        g_0 = g[:, 0]
        g_1 = g[:, 1]
        idxs=random.sample(range(len(g)), num_x)
        g_0=[g_0[i] for i in idxs]
        g_1=[g_1[i] for i in idxs]
        yield features[g_0], g_1
        

                
                
                    
                


