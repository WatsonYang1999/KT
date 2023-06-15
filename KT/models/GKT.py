import math
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            # batch_size == 1 or 0 will cause BatchNorm error, so return the input directly
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:  # len(input_size()) == 2
            return self.norm(inputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class EraseAddGate(nn.Module):
    def __init__(self, feature_dim, concept_num, bias=True):
        super(EraseAddGate, self).__init__()

        self.weight = nn.Parameter(torch.rand(concept_num))
        self.reset_parameters()
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        erase_gate = torch.sigmoid(self.erase(x))

        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))
        return tmp_x + self.weight.unsqueeze(dim=1) * add_feat

class GKT(nn.Module):

    def __init__(self, concept_num, hidden_dim, embedding_dim, edge_type_num, graph_type, graph=None, graph_model=None,
                 dropout=0.5, bias=True, binary=False, has_cuda=False):
        super(GKT, self).__init__()
        self.concept_num = concept_num+1
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.edge_type_num = edge_type_num

        self.res_len = 2
        self.has_cuda = has_cuda

        assert graph_type in ['Dense', 'Transition', 'DKT', 'PAM', 'MHA', 'VAE']
        self.graph_type = graph_type
        if graph_type in ['Dense', 'Transition', 'DKT']:
            assert edge_type_num == 2
            assert graph is not None and graph_model is None
            self.graph = nn.Parameter(graph)  # [concept_num, concept_num]
            self.graph.requires_grad = False  # fix parameter
            self.graph_model = graph_model
        else:  # ['PAM', 'MHA', 'VAE']
            assert graph is None
            self.graph = graph  # None
            if graph_type == 'PAM':
                assert graph_model is None
                self.graph = nn.Parameter(torch.rand(concept_num, concept_num))
            else:
                assert graph_model is not None
            self.graph_model = graph_model

        # one-hot feature and question

        one_hot_feat = torch.eye(self.res_len * self.concept_num)
        print(one_hot_feat.shape)
        self.one_hot_feat = one_hot_feat.cuda() if self.has_cuda else one_hot_feat

        self.one_hot_q = torch.eye(self.concept_num, device=self.one_hot_feat.device)

        zero_padding = torch.zeros(1, self.concept_num, device=self.one_hot_feat.device)

        self.one_hot_q = torch.cat((self.one_hot_q, zero_padding), dim=0)
        # concept and concept & response embeddings
        self.emb_x = nn.Embedding(self.res_len * concept_num, embedding_dim)
        # last embedding is used for padding, so dim + 1
        self.emb_c = nn.Embedding(concept_num + 1, embedding_dim, padding_idx=-1)

        # f_self function and f_neighbor functions
        mlp_input_dim = hidden_dim + embedding_dim
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.f_neighbor_list = nn.ModuleList()
        if graph_type in ['Dense', 'Transition', 'DKT', 'PAM']:
            # f_in and f_out functions
            self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
            self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
        else:  # ['MHA', 'VAE']
            for i in range(edge_type_num):
                self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))

        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(hidden_dim, concept_num)
        # Gate Recurrent Unit
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        # prediction layer
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

    # Aggregate step, as shown in Section 3.2.1 of the paper
    def _aggregate(self, xt, qt, ht, batch_size):
        r"""
        Parameters:
            xt: input one-hot question answering features at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
            ht: hidden representations of all concepts at the current timestamp
            batch_size: the size of a student batch
        Shape:
            xt: [batch_size]
            qt: [batch_size]
            ht: [batch_size, concept_num, hidden_dim]
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
        Return:
            tmp_ht: aggregation results of concept hidden knowledge state and concept(& response) embedding
        """

        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        x_idx_mat = torch.arange(self.res_len * self.concept_num, device=xt.device)
        print(x_idx_mat)
        x_embedding = self.emb_x(x_idx_mat)  # [res_len * concept_num, embedding_dim]
        masked_feat = F.embedding(xt[qt_mask], self.one_hot_feat)  # [mask_num, res_len * concept_num]
        res_embedding = masked_feat.mm(x_embedding)  # [mask_num, embedding_dim]
        mask_num = res_embedding.shape[0]

        concept_idx_mat = self.concept_num * torch.ones((batch_size, self.concept_num), device=xt.device).long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.concept_num, device=xt.device)
        concept_embedding = self.emb_c(concept_idx_mat)  # [batch_size, concept_num, embedding_dim]

        index_tuple = (torch.arange(mask_num, device=xt.device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)  # [batch_size, concept_num, hidden_dim + embedding_dim]
        return tmp_ht

    # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
    def _agg_neighbors(self, tmp_ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            qt: [batch_size]
            m_next: [batch_size, concept_num, hidden_dim]
        Return:
            m_next: hidden representations of all concepts aggregating neighboring representations at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        masked_qt = qt[qt_mask]  # [mask_num, ]
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, concept_num, hidden_dim + embedding_dim]
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_dim + embedding_dim]
        self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.concept_num,
                                                           1)  # [mask_num, concept_num, hidden_dim + embedding_dim]
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht),
                             dim=-1)  # [mask_num, concept_num, 2 * (hidden_dim + embedding_dim)]
        concept_embedding, rec_embedding, z_prob = None, None, None

        if self.graph_type in ['Dense', 'Transition', 'DKT', 'PAM']:
            adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
            reverse_adj = self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(
                dim=-1)  # [mask_num, concept_num, 1]
            # self.f_neighbor_list[0](neigh_ht) shape: [mask_num, concept_num, hidden_dim]
            neigh_features = adj * self.f_neighbor_list[0](neigh_ht) + reverse_adj * self.f_neighbor_list[1](neigh_ht)
        else:  # ['MHA', 'VAE']
            concept_index = torch.arange(self.concept_num, device=qt.device)
            concept_embedding = self.emb_c(concept_index)  # [concept_num, embedding_dim]
            if self.graph_type == 'MHA':
                query = self.emb_c(masked_qt)
                key = concept_embedding
                att_mask = Variable(torch.ones(self.edge_type_num, mask_num, self.concept_num, device=qt.device))
                for k in range(self.edge_type_num):
                    index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
                    att_mask[k] = att_mask[k].index_put(index_tuple, torch.zeros(mask_num, device=qt.device))
                graphs = self.graph_model(masked_qt, query, key, att_mask)
            else:  # self.graph_type == 'VAE'
                sp_send, sp_rec, sp_send_t, sp_rec_t = self._get_edges(masked_qt)
                graphs, rec_embedding, z_prob = self.graph_model(concept_embedding, sp_send, sp_rec, sp_send_t,
                                                                 sp_rec_t)
            neigh_features = 0
            for k in range(self.edge_type_num):
                adj = graphs[k][masked_qt, :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
                if k == 0:
                    neigh_features = adj * self.f_neighbor_list[k](neigh_ht)
                else:
                    neigh_features = neigh_features + adj * self.f_neighbor_list[k](neigh_ht)
            if self.graph_type == 'MHA':
                neigh_features = 1. / self.edge_type_num * neigh_features
        # neigh_features: [mask_num, concept_num, hidden_dim]
        m_next = tmp_ht[:, :, :self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next, concept_embedding, rec_embedding, z_prob

    # Update step, as shown in Section 3.3.2 of the paper
    def _update(self, tmp_ht, ht, qt):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            ht: hidden representations of all concepts at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            ht: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            h_next: [batch_size, concept_num, hidden_dim]
        Return:
            h_next: hidden representations of all concepts at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        mask_num = qt_mask.nonzero().shape[0]
        # GNN Aggregation
        m_next, concept_embedding, rec_embedding, z_prob = self._agg_neighbors(tmp_ht,
                                                                               qt)  # [batch_size, concept_num, hidden_dim]
        # Erase & Add Gate
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, concept_num, hidden_dim]
        # GRU
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim),
                       ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num * concept_num, hidden_num]
        index_tuple = (torch.arange(mask_num, device=qt_mask.device),)
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.concept_num, self.hidden_dim))
        return h_next, concept_embedding, rec_embedding, z_prob

    # Predict step, as shown in Section 3.3.3 of the paper
    def _predict(self, h_next, qt):
        r"""
        Parameters:
            h_next: hidden representations of all concepts at the next timestamp after the update step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            h_next: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            y: [batch_size, concept_num]
        Return:
            y: predicted correct probability of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, concept_num]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, concept_num]
        return y

    def _get_next_pred(self, yt, q_next):
        r"""
        Parameters:
            yt: predicted correct probability of all concepts at the next timestamp
            q_next: question index matrix at the next timestamp
            batch_size: the size of a student batch
        Shape:
            y: [batch_size, concept_num]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        next_qt = q_next
        next_qt = torch.where(next_qt != -1, next_qt, self.concept_num * torch.ones_like(next_qt, device=yt.device))
        one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)  # [batch_size, concept_num]
        # dot product between yt and one_hot_qt
        pred = (yt * one_hot_qt).sum(dim=1)  # [batch_size, ]
        return pred

    # Get edges for edge inference in VAE
    def _get_edges(self, masked_qt):
        r"""
        Parameters:
            masked_qt: qt index with -1 padding values removed
        Shape:
            masked_qt: [mask_num, ]
            rel_send: [edge_num, concept_num]
            rel_rec: [edge_num, concept_num]
        Return:
            rel_send: from nodes in edges which send messages to other nodes
            rel_rec:  to nodes in edges which receive messages from other nodes
        """
        mask_num = masked_qt.shape[0]
        row_arr = masked_qt.cpu().numpy().reshape(-1, 1)  # [mask_num, 1]
        row_arr = np.repeat(row_arr, self.concept_num, axis=1)  # [mask_num, concept_num]
        col_arr = np.arange(self.concept_num).reshape(1, -1)  # [1, concept_num]
        col_arr = np.repeat(col_arr, mask_num, axis=0)  # [mask_num, concept_num]
        # add reversed edges
        new_row = np.vstack((row_arr, col_arr))  # [2 * mask_num, concept_num]
        new_col = np.vstack((col_arr, row_arr))  # [2 * mask_num, concept_num]
        row_arr = new_row.flatten()  # [2 * mask_num * concept_num, ]
        col_arr = new_col.flatten()  # [2 * mask_num * concept_num, ]
        data_arr = np.ones(2 * mask_num * self.concept_num)
        init_graph = sp.coo_matrix((data_arr, (row_arr, col_arr)), shape=(self.concept_num, self.concept_num))
        init_graph.setdiag(0)  # remove self-loop edges
        row_arr, col_arr, _ = sp.find(init_graph)
        row_tensor = torch.from_numpy(row_arr).long()
        col_tensor = torch.from_numpy(col_arr).long()
        one_hot_table = torch.eye(self.concept_num, self.concept_num)
        rel_send = F.embedding(row_tensor, one_hot_table)  # [edge_num, concept_num]
        rel_rec = F.embedding(col_tensor, one_hot_table)  # [edge_num, concept_num]
        sp_rec, sp_send = rel_rec.to_sparse(), rel_send.to_sparse()
        sp_rec_t, sp_send_t = rel_rec.T.to_sparse(), rel_send.T.to_sparse()
        sp_send = sp_send.to(device=masked_qt.device)
        sp_rec = sp_rec.to(device=masked_qt.device)
        sp_send_t = sp_send_t.to(device=masked_qt.device)
        sp_rec_t = sp_rec_t.to(device=masked_qt.device)
        return sp_send, sp_rec, sp_send_t, sp_rec_t

    def forward(self, features, questions):
        r"""
        Parameters:
            features: input one-hot matrix
            questions: question index matrix
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            features: [batch_size, seq_len]
            questions: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        batch_size, seq_len = features.shape
        ht = Variable(torch.zeros((batch_size, self.concept_num, self.hidden_dim), device=features.device))
        pred_list = []
        ec_list = []  # concept embedding list in VAE
        rec_list = []  # reconstructed embedding list in VAE
        z_prob_list = []  # probability distribution of latent variable z in VAE

        for i in range(seq_len):
            xt = features[:, i]  # [batch_size]
            qt = questions[:, i]  # [batch_size]
            qt_mask = torch.ne(qt, -1)  # [batch_size], next_qt != -1

            tmp_ht = self._aggregate(xt, qt, ht, batch_size)  # [batch_size, concept_num, hidden_dim + embedding_dim]

            h_next, concept_embedding, rec_embedding, z_prob = self._update(tmp_ht, ht,
                                                                            qt)  # [batch_size, concept_num, hidden_dim]

            ht[qt_mask] = h_next[qt_mask]  # update new ht
            yt = self._predict(h_next, qt)  # [batch_size, concept_num]

            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])

                pred_list.append(pred)
            ec_list.append(concept_embedding)
            rec_list.append(rec_embedding)
            z_prob_list.append(z_prob)

        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]
        return pred_res, ec_list, rec_list, z_prob_list

if __name__=='__main__':
    question_num = 100


    def build_dense_graph(node_num):
        graph = 1. / (node_num - 1) * np.ones((node_num, node_num))
        np.fill_diagonal(graph, 0)
        graph = torch.from_numpy(graph).float()
        return graph
    graph = build_dense_graph(question_num)
    hidden_dim = 100
    embedding_dim = 100
    dropout = 0.1
    graph_type = 'Dense'
    edge_types = 2
    graph_model = None
    model = GKT(question_num, hidden_dim, embedding_dim, edge_types, graph_type, graph=graph,
                graph_model=graph_model,
                dropout=dropout, has_cuda=torch.cuda.is_available())
    def count_model_parameters(model: torch.nn.Module):
        param_count = 0
        for param in model.parameters():
            param_count += param.nelement()

        for buffer in model.buffers():
            param_count += buffer.nelement()
        print('model parameter number: {:.3f}M'.format(param_count / 1000000))


    def get_model_size(model: torch.nn.Module):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('model size: {:.3f}MB'.format(size_all_mb))

    count_model_parameters(model)
    get_model_size(model)