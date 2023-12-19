import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.functional import F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from KT.models.DKT import  DKT_PLUS


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
    hidden_dim = 100
    output_dim = 100
    embed_dim = 100
    model = DKT_PLUS(
        s_num=s_num,
        q_num=q_num,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        embed_dim=embed_dim
    )
    model.set_qs_matrix()
