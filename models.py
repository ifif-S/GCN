import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttention

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init, alpha, nheads):
        super(GCN, self).__init__()

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


    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        
              
        x_gcn= torch.cat([F.dropout(F.relu(self.gc1(att(x, adj), adj)), self.dropout, training=self.training) for att in self.attentions], dim=1)
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        x_gcn = F.elu(self.gc2(self.out_att(x_gcn, adj), adj))
        
        
        '''
        x_gcn = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        
        x_gan = F.dropout(x_gcn, self.dropout, training=self.training)
        x_gan = torch.cat([att(x_gan, adj) for att in self.attentions], dim=1)
        x_gan = F.dropout(x_gan, self.dropout, training=self.training)
        x_gan= F.elu(self.out_att(x_gan, adj))
        x_gcn = self.gc2(x_gan, adj)        
        x_gan = F.log_softmax(x_gan, dim=1)
        '''
        '''
        
        
        x_gcn = self.gc2(x_gcn, adj)
        x_gcn = F.log_softmax(x_gcn, dim=1)
        
        
        x_gan = F.dropout(x, self.dropout, training=self.training)
        x_gan = torch.cat([att(x_gan, adj) for att in self.attentions], dim=1)
        x_gan = F.dropout(x_gan, self.dropout, training=self.training)
        x_gan= F.elu(self.out_att(x_gan, adj))
        x_gan = F.log_softmax(x_gan, dim=1)
        x = torch.cat((x_gcn, x_gan),dim=1)
        
        x = self.fc_gcn_gat(x)
        
        
        x = torch.cat((x_gcn, x_gan),dim=1)
        '''
        return F.log_softmax(x, dim=1)
        '''
        
        return F.log_softmax(x, dim=1)
        '''
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
