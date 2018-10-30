import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttention

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init, alpha, nheads):
        super(GCN, self).__init__()
        '''
        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        #self.gc2 = GraphConvolution(nhid, nhid, init=init)
        self.dropout = dropout
        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False) for _ in range(nheads)]
        
        #self.attentions = [GraphAttention(nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttention(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.fc_gcn_gat = nn.Linear(nclass*2, nclass)
        '''
        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)

class GCN_cont(nn.Module):
    def __init__(self, nfeat, nembed, nx, dropout, init, alpha, nheads):
        super(GCN_cont, self).__init__()
        
        self.gc1 = GraphConvolution(nfeat, nembed, init=init)
        self.gc2 = GraphConvolution(nembed, nx, init=init)
        self.dropout = dropout

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)
    def get_hiddenLayer(self):
        return self.gc1.weight

 
class GAT_cont(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,hiddenWeight, embedding_size):
        super(GAT_cont, self).__init__()
        self.dropout = dropout
        self.hiddenWeight = hiddenWeight
        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        
        self.nonLinear_x_2 = nn.Softmax()
        self.fc_x_2 = nn.Linear(embedding_size, nclass)
        self.nonLinear1_x_2 = nn.Softmax()
        self.fc2_x_2 = nn.Linear( nclass*2, nclass)

    def forward(self, x, adj):
        inputs = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        
        x_2 = torch.mm( inputs, self.hiddenWeight)
        x_2 = self.nonLinear_x_2(x_2)
        x_2 = self.fc_x_2(x_2)
        x_2 = self.nonLinear1_x_2(x_2)
        
        l_x = torch.cat((x, x_2),dim=1)
        l_x = self.fc2_x_2( l_x )
        
        
        
        
        return F.log_softmax(l_x, dim=1), x_2, x


class GCN_drop_in(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init):
        super(GCN_drop_in, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)
    


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
