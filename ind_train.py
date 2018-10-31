#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:00:08 2018

@author: ififsun
"""

import time
import random
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

from torch.autograd import Variable
from utils import *
from models import GCN, GAT, GCN_cont, GAT_cont
from opts import TrainOptions

"""
N : number of nodes
D : number of features per node
E : number of classes

@ input :
    - adjacency matrix (N x N)
    - feature matrix (N x D)
    - label matrix (N x E)

@ dataset :
    - citeseer
    - cora
    - pubmed
"""
opt = TrainOptions().parse()

# Data upload
adj, features, labels, idx_train, idx_val, idx_test, allx, y, x_size, graph = load_data(path=opt.dataroot, dataset=opt.dataset)
nclass = int(labels.max().item()) + 1


use_gpu = torch.cuda.is_available()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if use_gpu:
    torch.cuda.manual_seed(42)

model, optimizer = None, None
best_acc = 0

# Define the model and optimizer
if (opt.model == 'basic'):
    print("| Constructing basic GCN model...")
    model = GCN(
            nfeat = features.shape[1],
            nhid = opt.num_hidden,
            nclass = labels.max().item() + 1,
            dropout = opt.dropout,
            init = opt.init_type,
            nheads = opt.nb_heads,
            alpha = opt.alpha
    )
elif (opt.model == 'attention'):
    print("| Constructing Attention GCN model...")
    model = GAT(
            nfeat = features.shape[1],
            nhid = opt.num_hidden,
            nclass = int(labels.max().item()) + 1,
            dropout = opt.dropout,
            nheads = opt.nb_heads,
            alpha = opt.alpha
    )
elif (opt.model == 'semi-attention'):
    print("| Constructing Semi-Attention GCN model ...")
    model_gx = GCN_cont(
            nfeat = features.shape[1],
            nembed = opt.embedding_size,
            nx = adj.shape[0],
            dropout = opt.dropout_gcn,
            init = opt.init_type,
            nheads = opt.nb_heads,
            alpha = opt.alpha
    )
    model_x = GAT_cont(
            nfeat = features.shape[1],
            nhid = opt.num_hidden,
            nclass = int(labels.max().item()) + 1,
            dropout = opt.dropout,
            nheads = opt.nb_heads,
            alpha = opt.alpha,
            hiddenWeight = model_gx.get_hiddenLayer(),
            embedding_size = opt.embedding_size
            )
    
else:
    raise NotImplementedError

if (opt.optimizer == 'sgd'):
    optimizer = optim.SGD(
            model_x.parameters(),
            lr = opt.lr,
            weight_decay = opt.weight_decay,
            momentum = 0.9
    )
elif (opt.optimizer == 'adam'):
    optimizer = optim.Adam(
            model_x.parameters(),
            lr = opt.lr,
            weight_decay = opt.weight_decay
    )
else:
    raise NotImplementedError

if use_gpu:
    model.cuda()
    features, adj, labels, idx_train, idx_val, idx_test = \
        list(map(lambda x: x.cuda(), [features, adj, labels, idx_train, idx_val, idx_test]))

features, adj, labels = list(map(lambda x : Variable(x), [features, adj, labels]))

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

save_point = os.path.join('./checkpoint', opt.dataset)

if not os.path.isdir(save_point):
    os.mkdir(save_point)

def lr_scheduler(epoch, opt):
    return opt.lr * (0.5 ** (epoch / opt.lr_decay_epoch))

# Train
'''
def train(epoch):
    global best_acc

    t = time.time()
    model.train()
    optimizer.lr = lr_scheduler(epoch, opt)
    optimizer.zero_grad()

    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    # Validation for each epoch
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if acc_val > best_acc:
        best_acc = acc_val
        state = {
            'model': model,
            'acc': best_acc,
            'epoch': epoch,
        }

        torch.save(state, os.path.join(save_point, '%s.t7' %(opt.model)))

    sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.write("=> Training Epoch #{} : lr = {:.4f}".format(epoch, optimizer.lr))
    sys.stdout.write(" | Training acc : {:6.2f}%".format(acc_train.data.cpu().numpy() * 100))
    sys.stdout.write(" | Best acc : {:.2f}%". format(best_acc.data.cpu().numpy() * 100))
'''
def comp_iter(iter):
    """an auxiliary function used for computing the number of iterations given the argument iter.
    iter can either be an int or a float.
    """
    if iter >= 1:
        return iter
    return 1 if random.random() < iter else 0

def train(epoch):
    global best_acc
    optimizer.lr = lr_scheduler(epoch, opt)
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()

        
        
        
    output, hid_gcn, hid_gat = model_x(features, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    loss_train += F.cross_entropy ( hid_gcn[idx_train], labels[idx_train])
    loss_train += F.cross_entropy ( hid_gat[idx_train], labels[idx_train])
    
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()
    
    
    model_x.eval()
    output, hid_gcn, hid_gat = model_x(features, adj)

    acc_val = accuracy(output[idx_val], labels[idx_val])

    if acc_val > best_acc:
        best_acc = acc_val
        state = {
            'model': model_x,
            'acc': best_acc,
            'epoch': epoch,
        }

        torch.save(state, os.path.join(save_point, '%s.t7' %(opt.model)))

    sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.write("=> Training Epoch #{} : lr = {:.4f}".format(epoch, optimizer.lr))
    sys.stdout.write(" | Training acc : {:6.2f}%".format(acc_train.data.cpu().numpy() * 100))
    sys.stdout.write(" | Best acc : {:.2f}%". format(best_acc.data.cpu().numpy() * 100))

def init_train():
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_gx.parameters(), lr=opt.graph_learning_rate)
    optimizer.zero_grad()
    
    

    for i in range(opt.init_iter_graph):
        gx, gy = next(gen_graph(graph, features))
        l_gx = model_gx(gx, adj)
        g_loss = criterion(l_gx, torch.LongTensor(gy))
    g_loss.backward()
    optimizer.step()
    print ('iter label', i, g_loss)
    

# Main code for training
if __name__ == "__main__":
    print("\n[STEP 2] : Obtain (adjacency, feature, label) matrix")
    print("| Adjacency matrix : {}".format(adj.shape))
    print("| Feature matrix   : {}".format(features.shape))
    print("| Label matrix     : {}".format(labels.shape))
    init_train()
    # Training
    print("\n[STEP 3] : Training")
    for epoch in range(1, opt.epoch+1):
        train(epoch)
    print("\n=> Training finished!")
