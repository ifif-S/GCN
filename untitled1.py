#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 21:53:28 2018

@author: ififsun
"""

       '''
    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        
              
        x_gcn= torch.cat([F.dropout(F.relu(self.gc1(att(x, adj), adj)), self.dropout, training=self.training) for att in self.attentions], dim=1)
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        x_gcn = F.elu(self.gc2(self.out_att(x_gcn, adj), adj))
        '''
        
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
        
        #return F.log_softmax(x, dim=1)