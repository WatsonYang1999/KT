import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.functional import F
from torch.nn.parameter import Parameter
from gcn import GCN

class QGKT(nn.Module):
    def __init__(self, skill_num, question_num, hidden_dim, embedding_dim, qs_matrix, s_graph):
        super(QGKT, self).__init__()
        print(f'Skill Num: {skill_num}')
        print(f'Question Num: {question_num}')

        self.skill_num = skill_num
        self.question_num = question_num
        self.feature_dim = question_num + 1
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        '''
        qs_matrix[i,j] = 1 indicates that q_i is related to s_j,
        qs_matrix[q_n,:] = 0 cuz it represents the padding value  
        '''
        #qs_matrix = np.concatenate((np.zeros((1, self.skill_num)), qs_matrix))
        #self.sq_matrix = torch.sparse.Tensor(qs_matrix.transpose())
        #self.qs_matrix = torch.sparse.Tensor(qs_matrix)  # [q_num,s_num]
        #print(s_graph)
        #self.s_graph = torch.sparse.Tensor(s_graph)
        # self.attention = nn.MultiheadAttention()
        '''
        这里的q和s的两个embedding层的向量维度应该一样吗？
        '''
        self.q_embedding = nn.Embedding(self.feature_dim, self.embedding_dim, padding_idx=-1)
        self.s_embedding = nn.Embedding(self.skill_num * 2, self.embedding_dim, padding_idx=-1)
        self.a_embedding = nn.Embedding(2, self.embedding_dim, padding_idx=-1)
        self.gru = nn.GRUCell(int(embedding_dim), hidden_dim)
        # self.gru = nn.GRUCell(int(embedding_dim) * skill_num, hidden_dim * skill_num)
        self.gru_list = [nn.GRUCell(embedding_dim, hidden_dim) for i in range(0, skill_num)]

        self.graph_conv = GCN()
        self.predict_layer = nn.Linear(self.hidden_dim, 1)
        self.simple_predict_v2 = nn.Linear(self.hidden_dim, self.question_num + 1, bias=True)
        self.mlp_layer = nn.Linear(self.skill_num, self.question_num + 1)
        self.mask_weight = Parameter(torch.FloatTensor(self.skill_num, self.question_num + 1))
        self.mask_bias = Parameter(torch.FloatTensor(self.question_num + 1))
        stdv = 1. / math.sqrt(self.mask_weight.size(1))
        self.mask_weight.data.uniform_(-stdv, stdv)
        if self.mask_bias is not None:
            self.mask_bias.data.uniform_(-stdv, stdv)

    def prob_multiply_predict(self, ht, qt):
        '''

        :param ht: [batch_size,skill_num,hidden_dim]
        :param qt: [batch_size] : the relevant question number
        :return:
        '''

        skill_master_prob = torch.sigmoid(self.predict_layer(ht)).squeeze()  # [batch_size,skill_num,1]

        # [batch_size,skill_num] @ [skill_num,question_num]

        question_master_prob = torch.log(skill_master_prob) @ self.qs_matrix.transpose(0, 1)

        question_master_prob = torch.exp(question_master_prob)  # [batch_size,question_num]

        mask_selected = F.one_hot(qt, num_classes=self.question_num + 1)

        question_master_prob_selected = (question_master_prob * mask_selected) @ torch.ones([self.question_num + 1, 1])
        return question_master_prob_selected

    def predict(self, ht, qt):
        '''

        :param ht: [batch_size,skill_num,hidden_dim]
        :param qt: [batch_size] : the relevant question number
        :return:
        '''

        skill_master_prob = torch.sigmoid(self.predict_layer(ht)).squeeze()  # [batch_size,skill_num]

        # [batch_size,skill_num] @ [skill_num,question_num]

        question_master_prob = torch.log(skill_master_prob) @ self.qs_matrix.transpose(0, 1)

        question_master_prob = torch.exp(question_master_prob)  # [batch_size,question_num]

        mask_selected = F.one_hot(qt, num_classes=self.question_num + 1)

        question_master_prob_selected = (question_master_prob * mask_selected) @ torch.ones([self.question_num + 1, 1])

        return question_master_prob_selected

    def simple_mlp_predict(self, ht: torch.Tensor, qt: torch.Tensor):
        '''

        :param ht: [batch_size,skill_num,hidden_dim]
        :param qt:
        :return:
        '''

        ht = ht.transpose(1, 2)

        ht = self.mlp_layer(ht)

        ht = ht.transpose(1, 2)

        question_master_prob = torch.sigmoid(self.predict_layer(ht)).squeeze()

        mask_selected = F.one_hot(qt, num_classes=self.question_num + 1)

        question_master_prob_selected = (question_master_prob * mask_selected) @ torch.ones([self.question_num + 1, 1])
        return question_master_prob_selected

    def masked_mlp_predict(self, ht: torch.Tensor, qt: torch.Tensor):
        '''

        :param ht: [batch_size,skill_num,hidden_dim]
        :param qt:
        :return:
        '''
        device = ht.device
        ht = ht.transpose(1, 2)
        mask_weight = (self.mask_weight * self.sq_matrix.to(ht.device))
        ht = ht @ mask_weight + self.mask_bias
        ht = ht.transpose(1, 2)

        question_master_prob = torch.sigmoid(self.predict_layer(ht)).squeeze()

        mask_selected = F.one_hot(qt, num_classes=self.question_num + 1)

        question_master_prob_selected = (question_master_prob * mask_selected) @ torch.ones(
            [self.question_num + 1, 1]).to(device)
        return question_master_prob_selected

    def _aggregate(self, qt):
        pass

    def forward_v2(self, features: torch.Tensor, questions: torch.Tensor, answers: torch.Tensor):
        device = features.device
        features = torch.where(features < 0, torch.full(features.shape, fill_value=0).long().to(device), features)
        questions = torch.where(questions < 0, torch.full(questions.shape, fill_value=0).long().to(device), questions)
        answers = torch.where(answers < 0, torch.full(answers.shape, fill_value=0).long().to(device), answers)

        batch_size, seq_len = features.shape
        predict_total = []
        ht = Variable(torch.zeros((batch_size, self.hidden_dim), device=device))
        for i in range(seq_len):
            qt = questions[:, i]
            at = answers[:, i]
            qt_embedding = self.q_embedding(qt)
            at_embedding = self.a_embedding(at)
            aggregated = qt_embedding + at_embedding

            ht = self.gru(aggregated, ht)
            question_master_prob = torch.sigmoid(self.simple_predict_v2(ht))

            question_select_matrix = F.embedding(input=qt, weight=torch.eye(self.question_num + 1).to(device)).to(
                device)
            assert question_select_matrix.shape == question_master_prob.shape
            # [batch_size,question_num] *[batch_size,question_num] @ [question_num,1] = [batch_size,1]

            question_master_prob_selected = (question_master_prob * question_select_matrix)

            wtf = torch.ones([self.question_num + 1, 1])
            assert question_master_prob_selected.shape[1] == wtf.shape[0]
            question_master_prob_selected = question_master_prob_selected @ wtf.to(device)
            predict_total.append(question_master_prob_selected)

        predict_total = torch.stack(predict_total)

        predict_total = predict_total.transpose(0, 1).squeeze()

        check = (predict_total == predict_total)
        assert torch.all(check == torch.tensor(True))

        if len(predict_total.shape) < 2:
            predict_total = predict_total.reshape([1,-1])
        return predict_total[:, :-1]

    def forward_v1(self, features: torch.Tensor, questions: torch.Tensor, answers: torch.Tensor):
        device = features.device

        '''
        Replace the padding elements with value 0
        '''
        features = torch.where(features < 0, torch.full(features.shape, fill_value=0).long().to(device), features)
        questions = torch.where(questions < 0, torch.full(questions.shape, fill_value=0).long().to(device), questions)
        answers = torch.where(answers < 0, torch.full(answers.shape, fill_value=0).long().to(device), answers)

        batch_size, seq_len = features.shape
        ht = Variable(torch.zeros((batch_size, self.skill_num, self.hidden_dim), device=device))

        n_index = torch.zeros([batch_size, self.skill_num], dtype=torch.int64).to(device)  # [batch_size,s_n]
        for i in range(0, self.skill_num): n_index[:, i] = i
        predict_total = []
        for i in range(seq_len):
            qt = questions[:, i]
            at = answers[:, i]
            assert at.is_cuda
            xt = F.embedding(input=qt, weight=torch.eye(self.question_num + 1).to(device))  # [batch_size,q_n+1]
            xt = xt @ self.qs_matrix.to(device)  # [batch_size,s_n]
            st = n_index + at.unsqueeze(-1).expand(batch_size, self.skill_num) * self.skill_num
            assert st.is_cuda
            all_node_feature = self.s_embedding(st.to(device))  # [batch_size,s_n,emb_dim]

            selected_mask = xt.unsqueeze(-1).expand((batch_size, self.skill_num, self.embedding_dim))

            mask_node_feature = selected_mask * all_node_feature  # [batch_size,s_n,emb_dim]
            # graph convolution process : input : xt[batch_size,s_n,feature_dim]
            del selected_mask
            del st
            del all_node_feature
            '''
            小问题：这个RNN是吧所有节点的二维特征压缩成一维的，还是每个Node分开来算比较合适捏
            '''

            xt = self.graph_conv(mask_node_feature, self.s_graph.to(device))  # [batch_size,s_n,out_dim]
            #
            # xt = xt.reshape([batch_size,-1])
            #
            # ht = ht.reshape([batch_size,-1])

            xt = xt.reshape([-1, self.embedding_dim])

            ht = ht.reshape([-1, self.hidden_dim])

            ht = self.gru(xt, ht)

            ht = ht.reshape([batch_size, self.skill_num, -1])

            # question_master_prob = self.predict(ht, qt)  # [batch_size,1]

            question_master_prob = self.masked_mlp_predict(ht, qt)
            predict_total.append(question_master_prob)
        predict_total = torch.stack(predict_total)

        predict_total = predict_total.transpose(0, 1).squeeze()
        assert not predict_total.isnan().any()
        print(predict_total.shape)
        exit(-1)
        return predict_total[:, :-1]

    def forward(self, features: torch.Tensor, questions: torch.Tensor, answers: torch.Tensor):

        return self.forward_v2(features, questions, answers)


if __name__ == '__main__':
    import numpy as np
    import random
    q_num = 10000
    s_num = 100
    len_avg = 100
    len_max = 200
    batch_size = 128

    def seq_gen(seq_len: int):
        seq = []
        q_seq = np.random.randint(1, q_num + 1, size=[seq_len])
        a_seq = np.random.randint(0, 2, size=[seq_len])

        return q_seq, a_seq

    def data_gen():
        seq_len_arr = np.random.normal(len_avg, 5, batch_size).astype(np.int32)
        seq_len_arr = np.where(seq_len_arr < 0, 0 - seq_len_arr, seq_len_arr)
        seq_len_arr = np.where(seq_len_arr > len_max, seq_len_arr % len_max, seq_len_arr)
        X_fake = []
        y_fake = []
        for i in range(0, batch_size):
            q_seq, a_seq = seq_gen(seq_len_arr[i])
            q_seq = np.concatenate((q_seq, np.full([len_max - q_seq.shape[0]], -1)))
            a_seq = np.concatenate((a_seq, np.full([len_max - a_seq.shape[0]], -1)))
            X_fake.append(q_seq.tolist())
            y_fake.append(a_seq.tolist())

        return torch.LongTensor(np.array(X_fake)), torch.LongTensor(np.array(y_fake))

    def qs_matrix_gen(q_n, s_n):
        qs_matrix = np.zeros((q_n, s_n))
        sample_list = [i for i in range(0, s_n)]
        for i in range(0, q_n):
            random_skill_num = random.randint(1, s_n)
            skill_list = random.sample(sample_list, random_skill_num)

            for s in skill_list:
                qs_matrix[i, s] = 1

        # qs_matrix of shape [q_n,s_n]

        return qs_matrix

    def s_graph_gen(s_n):
        s_graph = np.ones((s_n, s_n))
        s_graph = s_graph / (s_n)
        np.fill_diagonal(s_graph, 1)

        return s_graph

    qs_matrix = qs_matrix_gen(q_num, s_num)
    # qs_matrix = np.load('../Dataset/Buaa2019s/qs_matrix.npy').transpose()
    gkt = QGKT(question_num=q_num, skill_num=s_num, hidden_dim=50, embedding_dim=100,
               qs_matrix=qs_matrix, s_graph=s_graph_gen(s_num))
    from util.kt_util import get_model_size
    from KT.models.Loss import KTLoss
    get_model_size(gkt)

    for i in range(0, 1000):
        q, a = data_gen()
        f = torch.where(a == 0, q + q_num, q)
        y_hat = gkt.forward(f, q, a)

        criterion = KTLoss()
        print(a)
        print(y_hat)
        loss, auc, acc = criterion(y_hat, a)
        print(f'Loss : {loss}')
        loss.backward()
