import random

import torch
from torch import nn
import numpy as np
class DKVMNHeadGroup(nn.Module):
    def __init__(self, memory_size, memory_state_dim, is_write):
        super(DKVMNHeadGroup, self).__init__()
        """"
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        if self.is_write:
            self.erase = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            self.add = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            nn.init.kaiming_normal_(self.erase.weight)
            nn.init.kaiming_normal_(self.add.weight)
            nn.init.constant_(self.erase.bias, 0)
            nn.init.constant_(self.add.bias, 0)

    def addressing(self, control_input, memory,mask=None):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
            mask :                  Shape (batch_size, memory_size)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """

        similarity_score = torch.matmul(control_input, torch.t(memory)) # [b， m] ,for every q in batch ,cal similarity  with memory slot
        if mask is not None:
            similarity_score = similarity_score * mask
        assert not torch.any(torch.isnan(similarity_score))
        correlation_weight = torch.nn.functional.softmax(similarity_score, dim=1) # Shape: (batch_size, memory_size)
        return correlation_weight

    def read(self, memory, control_input=None, read_weight=None):
        """
        Parameters
            control_input:  Shape (batch_size, control_state_dim)
            memory:         Shape (batch_size, memory_size, memory_state_dim)
            read_weight:    Shape (batch_size, memory_size)
        Returns
            read_content:   Shape (batch_size,  memory_state_dim)
        """

        if read_weight is None:
            read_weight = self.addressing(control_input=control_input, memory=memory)

        read_weight = read_weight.view(-1, 1)
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)

        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)

        read_content = torch.sum(read_content, dim=1)

        return read_content

    def write(self, control_input, memory, write_weight=None):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write
        if write_weight is None:
            write_weight = self.addressing(control_input=control_input, memory=memory)
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mult = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        new_memory = memory * (1 - erase_mult) + add_mul
        return new_memory


class DKVMN_Module(nn.Module):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key):
        super(DKVMN_Module, self).__init__()
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_value_state_dim)
        :param init_memory_value:       Shape (batch_size, memory_size, memory_value_state_dim)
        """
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.key_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                       memory_state_dim=self.memory_key_state_dim,
                                       is_write=False)

        self.value_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                         memory_state_dim=self.memory_value_state_dim,
                                         is_write=True)

        self.memory_key = init_memory_key

        # self.memory_value = self.init_memory_value
        self.memory_value = None

    def init_value_memory(self, memory_value):
        self.memory_value = memory_value

    def attention(self, control_input,mask=None):
        correlation_weight = self.key_head.addressing(control_input=control_input, memory=self.memory_key,mask=mask)

        return correlation_weight

    def set_qs_matrix(self, qs_matrix):
        self.qs_matrix = qs_matrix

    def read(self, read_weight):
        read_content = self.value_head.read(memory=self.memory_value, read_weight=read_weight)

        return read_content

    def write(self, write_weight, control_input, if_write_memory):
        memory_value = self.value_head.write(control_input=control_input,
                                             memory=self.memory_value,
                                             write_weight=write_weight)
        # if_write_memory = torch.cat([if_write_memory.unsqueeze(1) for _ in range(self.memory_value_state_dim)], 1)

        self.memory_value = nn.Parameter(memory_value.data)

        return self.memory_value


class DKVMN(nn.Module):

    def __init__(self, n_question, batch_size, q_embed_dim, qa_embed_dim,
                 memory_size, memory_key_state_dim, memory_value_state_dim, final_fc_dim, student_num=None):
        super(DKVMN, self).__init__()
        self.n_question = n_question
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.final_fc_dim = final_fc_dim
        self.student_num = student_num

        self.input_embed_linear = nn.Linear(self.q_embed_dim, self.final_fc_dim, bias=True)

        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.final_fc_dim, self.final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)
        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, self.memory_key_state_dim))
        nn.init.kaiming_normal_(self.init_memory_key)
        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal_(self.init_memory_value)

        self.mem = DKVMN_Module(memory_size=self.memory_size,
                                memory_key_state_dim=self.memory_key_state_dim,
                                memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

        # add problem diff layer for experiment
        self.difficult_param = nn.Embedding(3162+ 1, 1)
        self.q_embed_diff = nn.Embedding(self.n_question + 1, q_embed_dim)
        self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, qa_embed_dim)

    def get_hyperparameters(self):
        hyperparameters = {
            'n_question': self.n_question,
            'q_embed_dim': self.q_embed_dim,
            'memory_size': self.memory_size,
            'memory_key_state_dim': self.memory_key_state_dim,
            'memory_value_state_dim': self.memory_value_state_dim,
            'final_fc_dim': self.final_fc_dim,
        }
        return hyperparameters

    def set_qs_matrix(self, qs_matrix):
        self.mem.set_qs_matrix(qs_matrix)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)
        # nn.init.constant(self.input_embed_linear.bias, 0)
        # nn.init.normal(self.input_embed_linear.weight, std=0.02)

    def init_embeddings(self):

        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)


    def forward(self, q_data, qa_data, target,pid_data=None,student_id=None):

        seqlen = q_data.shape[1]
        batch_size = q_data.shape[0]
        target_2d = target.reshape(batch_size,-1)
        target_2d_is_pad = (target_2d<0)
        qa_data_is_pad = (qa_data==0)
        q_data_is_pad  = (q_data==0)
        # assert torch.all(q_data_is_pad == qa_data_is_pad)
        #
        # assert torch.all(qa_data_is_pad == target_2d_is_pad)
        #
        #
        # assert torch.max(q_data)<self.q_embed.num_embeddings
        # assert torch.max(q_data) >= 0
        assert not torch.any(torch.isnan(self.q_embed.weight))
        q_embed_data = self.q_embed(q_data)   #[batch_size,seqlen,q_embed_dim]
        assert not torch.any(torch.isnan(q_embed_data))
        qa_embed_data = self.qa_embed(qa_data) #[batch_size,seqlen,qa_embed_dim]
        if pid_data is not None:

            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct

            pid_embed_data = self.difficult_param(pid_data)
            assert q_embed_data.shape == q_embed_diff_data.shape

            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data  # uq *d_ct + c_ct
            qa_embed_diff_data = self.qa_embed_diff(
                qa_data)
            qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        slice_q_data = torch.chunk(q_data, seqlen, 1)

        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)


        value_read_content_l = []
        input_embed_l = []
        predict_logs = []
        for i in range(seqlen):
            ## Attention
            q = slice_q_embed_data[i].squeeze(1) #[b,d]
            assert not torch.any(torch.isnan(q))



            correlation_weight = self.mem.attention(q)


            assert not torch.any(torch.isnan(correlation_weight))
            if_memory_write = slice_q_data[i].squeeze(1).ge(1)
            if_memory_write = if_memory_write.float()
            ###if_memory_write = utils.varible(torch.FloatTensor(if_memory_write.data.tolist()), 1)

            ## Read Process
            read_content = self.mem.read(correlation_weight)

            assert not torch.any(torch.isnan(read_content))
            value_read_content_l.append(read_content)
            input_embed_l.append(q)
            ## Write Process
            qa = slice_qa_embed_data[i].squeeze(1)

            new_memory_value = self.mem.write(correlation_weight, qa, if_memory_write)

            # read_content_embed = torch.tanh(self.read_embed_linear(torch.cat([read_content, q], 1)))
            # pred = self.predict_linear(read_content_embed)
            # predict_logs.append(pred)

        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)

        # input_embed_content = input_embed_content.view(batch_size * seqlen, -1)
        # input_embed_content = torch.tanh(self.input_embed_linear(input_embed_content))
        # input_embed_content = input_embed_content.view(batch_size, seqlen, -1)

        predict_input = torch.cat([all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size*seqlen, -1)))

        pred = self.predict_linear(read_content_embed)


        # predicts = torch.cat([predict_logs[i] for i in range(seqlen)], 1)
        target_1d = target.view(-1,1)                   # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)               # [batch_size * seq_len, 1]
        # pred_1d = predicts.view(-1, 1)           # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        filtered_target = filtered_target.float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target

    def _forward_re(self,q_data,qa_data,target):
        bs = q_data.shape[0]
        seq_len = q_data.shape[1]
        q_embed = self.q_embed(q_data)
        qa_embed = self.qa_embed(qa_data)
        q_embed_chunk = torch.chunk(q_embed,chunks=seq_len,dim=1)  # chunk by the seq_len dim
        qa_embed_chunk = torch.chunk(qa_embed,chunks=seq_len,dim=1)
        q_seq_chunk = torch.chunk(qa_embed,chunks=seq_len,dim=1)

        read_value_list = []
        for i in range(seq_len):
            q_embed_i = q_embed_chunk[i].squeeze(1)
            qa_embed_i = qa_embed_chunk[i].squeeze(1)
            q_i = q_seq_chunk[i].squeeze(1)

            # get attention score

            correlation_weight = self.mem.attention(q_embed_i)

            # read
            read_value = self.mem.read(correlation_weight)

            read_value_list.append(read_value)

            # write
            if_memory_write = q_i.squeeze(1).ge(1)
            if_memory_write = if_memory_write.float()
            self.mem.write(correlation_weight, qa_embed_i, if_memory_write)

        batch_read_value_all = torch.cat([read_value_list[i].unsqueeze(1) for i in range(max_seq_len)], 1)


        predict_input = torch.cat([batch_read_value_all, q_embed], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(bs*max_seq_len, -1)))

        pred = self.predict_linear(read_content_embed)


        # predicts = torch.cat([predict_logs[i] for i in range(seqlen)], 1)
        target_1d = target.view(-1,1)                   # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)               # [batch_size * seq_len, 1]
        # pred_1d = predicts.view(-1, 1)           # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        filtered_target = filtered_target.float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target

if __name__=='__main__':
    embed_dim = 200
    q_embed_dim = 50
    qa_embed_dim = 200
    final_fc_dim = 50
    pro_num = 10000
    memory_size = 100
    bs = 256
    max_seq_len = 200

    model = DKVMN(n_question=pro_num,
                  batch_size=bs,
                  q_embed_dim=q_embed_dim,
                  qa_embed_dim=qa_embed_dim,
                  memory_size=20,
                  memory_key_state_dim=q_embed_dim,
                  memory_value_state_dim=qa_embed_dim,
                  final_fc_dim=final_fc_dim)

    count_model_parameters(model)
    get_model_size(model)

    def gen_dummy_data(q_num,s_num,bs = 64, max_seq_len = 200):
        fake_q_seq = torch.randint(low=0,high=q_num,size = (bs,max_seq_len))
        fake_a_seq = torch.randint(low=0,high=1,size = (bs,max_seq_len))
        fake_f_seq = fake_q_seq + fake_a_seq * q_num
        seq_len_list = [random.randint(int(max_seq_len/3),max_seq_len) for _ in range(0,bs)]
        for idx,l in enumerate(seq_len_list):
            fake_q_seq[idx,l:] = 0
            fake_a_seq[idx,l:] = -1
            fake_f_seq[idx,l:] = 0
        fake_qs_matrix = torch.randint(low=0,high=1,size=(q_num,s_num))
        return fake_f_seq,fake_q_seq, fake_a_seq, fake_qs_matrix
    f_seq, q_seq , a_seq,qs_matrix = gen_dummy_data(
        q_num=pro_num ,
        s_num=memory_size,
        bs = bs,
        max_seq_len=max_seq_len
    )
    loss1, pred1,target1 = model.forward(
        q_data=q_seq,
        qa_data=f_seq,
        target=a_seq
    )

    loss2, pred2,target2 = model.forward_re(
        q_data=q_seq,
        qa_data=f_seq,
        target=a_seq
    )


    print(loss1)
    print(loss2)