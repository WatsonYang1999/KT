import torch
import torch.nn as nn
from torch.nn import Parameter

class GraphConvolution(nn.Module):
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
        nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input_feat, node_indices, graph_adj_matrix):
        '''
        input_feat: [batch_size, seq_len, feature]: node feature, the input node may be duplicated and unranked
        node_indices: [batch_size, seq_len], the node idx in the graph
        graph_adj_matrix: [node_num, node_num], graph adjacency matrix, 1 for edge 0 for no edge
        '''

        # Flatten the batch and sequence dimensions for easy matrix multiplication
        batch_size, seq_len, feature_dim = input_feat.size()
        input_feat_flat = input_feat.view(batch_size * seq_len, feature_dim)
        node_indices_flat = node_indices.view(batch_size * seq_len)

        # Extract node features based on indices
        selected_node_feat = input_feat_flat[torch.arange(batch_size * seq_len), node_indices_flat]

        # Perform graph convolution
        support = torch.matmul(selected_node_feat, self.weight)

        # Aggregate information from neighboring nodes using the adjacency matrix
        aggregated_output = torch.matmul(graph_adj_matrix, support)

        # Reshape the aggregated output to [batch_size, seq_len, out_features]
        aggregated_output = aggregated_output.view(batch_size, seq_len, self.out_features)

        # Add bias if applicable
        if self.bias is not None:
            aggregated_output = aggregated_output + self.bias

        return aggregated_output

# Example usage:
# Assuming you have input features, node indices, and adjacency matrix
input_feat = torch.randn(2, 3, 4)  # Batch size of 2, sequence length of 3, feature dimension of 4
node_indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
graph_adj_matrix = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)

# Create GraphConvolution layer
in_features = 4
out_features = 8
gcn_layer = GraphConvolution(in_features, out_features)

# Forward pass
output = gcn_layer(input_feat, node_indices, graph_adj_matrix)
print(output)
