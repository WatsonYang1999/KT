import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.functional import F
# Load the Cora dataset (you may need to adjust the loading code based on your dataset format)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        '''
        input_feat: [batch_size,feature]: node feature,the input node may be duplicated and unranked
        node_indices: [batch_size], the node idx in the graph
        graph_adj_matrix: [batch_size,node_num], graph adjacency list for each node
        '''
        # support = torch.mm(input, self.weight)
        support = input @ self.weight #[bs,n_out] = [bs，n_in] * [n_in,n_out]

        output = adj @ support # [bs,n_out]= [bs，n_out] [bs,n_out]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_2layer(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN_2layer, self).__init__()
        print(f"nfeat :{nfeat}, nhid:{nhid}, nclass:{nout}")
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)
        return x


class GCN_1layer(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN_1layer, self).__init__()
        print(f"nfeat :{nfeat}, nhid:{nhid}, nclass:{nout}")
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


# Training loop
def train_model(model, data, optimizer, criterion, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            correct = pred[data.val_mask] == data.y[data.val_mask]
            accuracy = correct.sum().item() / data.val_mask.sum().item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val Accuracy: {accuracy:.4f}')


