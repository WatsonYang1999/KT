import numpy as np
import torch
from torch import nn
from KT.models.gcn import GCN_1layer
class SAKT(nn.Module):
    def __init__(self, q_num, seq_len, embed_dim, heads, dropout):
        super(SAKT, self).__init__()
        self.q_num = q_num
        self.seq_len = seq_len
        self.dim = embed_dim
        self.custom_feature = True
        self.embd_qa = nn.Embedding(2 * q_num + 1, embedding_dim=embed_dim)  # Interaction embedding
        self.embd_q = nn.Embedding(q_num + 1, embedding_dim=embed_dim)  # Excercise embedding
        self.embd_a = nn.Embedding(2 + 1, embedding_dim=embed_dim)
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=embed_dim)

        self.gcn = GCN_1layer(
            nfeat=embed_dim,
            nhid=embed_dim,
            nout=embed_dim,
            dropout=dropout)
        self.linear = nn.ModuleList(
            [nn.Linear(in_features=embed_dim, out_features=embed_dim) for x in
             range(3)])  # Linear projection for each embedding
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout)
        self.ffn = nn.ModuleList([nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True) for x in
                                  range(2)])  # feed forward layers post attention

        self.linear_out = nn.Linear(in_features=embed_dim, out_features=1, bias=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)  # output with correctnness prediction
        self.drop = nn.Dropout(dropout)

        self.graph = nn.Parameter(torch.eye(2 * self.q_num + 1))
        self.interaction_onehot = nn.Embedding(2*q_num+1,2*q_num+1)
        self.interaction_onehot.weight = nn.Parameter(
            #torch.eye(2*q_num+1),
            (1/(2 * self.q_num + 1))*torch.ones([2 * self.q_num + 1, 2 * self.q_num + 1]),
            requires_grad=False)
        print(self.interaction_onehot.weight)

    def forward_original(self, interaction, question, answer):
        interaction = torch.where(interaction < 0, torch.zeros_like(interaction), interaction)
        question = torch.where(question < 0, torch.zeros_like(question), question)

        device = interaction.device
        # positional embedding

        pos_index = torch.arange(self.seq_len).unsqueeze(0).to(device)

        pos_in = self.embd_pos(
            pos_index)  # making a tensor of 12 numbers, .unsqueeze(0) for converting to 2d, so as to get a 3d output

        # get the interaction embedding output

        out_in = self.embd_qa(interaction)  # (b, n) --> (b,n,d)

        out_in = out_in + pos_in

        # split the interaction embeding into v and k ( needs to verify if it is slpited or not)
        value_in = out_in
        key_in = out_in  # print('v,k ', value_in.shape)

        ## get the excercise embedding output

        query_ex = self.embd_q(question)  # (b,n) --> (b,n,d) #print(query_ex.shape)

        ## Linearly project all the embedings
        value_in = self.linear[0](value_in).permute(1, 0, 2)  # (b,n,d) --> (n,b,d)
        key_in = self.linear[1](key_in).permute(1, 0, 2)
        query_ex = self.linear[2](query_ex).permute(1, 0, 2)

        ## pass through multihead attention
        attn_mask = torch.from_numpy(
            np.triu(np.ones((self.seq_len, self.seq_len)), k=1).astype(
                'bool')).to(device)
        atn_out, _ = self.attn(query_ex, key_in, value_in,
                               attn_mask=attn_mask)  # lower triangular mask, bool, torch    (n,b,d)
        atn_out = query_ex + atn_out  # Residual connection ; added excercise embd as residual because previous ex may have imp info, suggested in paper.
        atn_out = self.layer_norm1(
            atn_out)  # Layer norm                        #print('atn',atn_out.shape) #n,b,d = atn_out.shape

        # take batch on first axis
        atn_out = atn_out.permute(1, 0, 2)  # (n,b,d) --> (b,n,d)

        ## FFN 2 layers
        # ffn_out = self.drop(
        #     self.ffn[1](
        #         nn.ReLU()
        #         (self.ffn[0](atn_out)
        #                   )
        #     )
        # )
        ffn_out = nn.ReLU()(self.ffn[0](atn_out))

        ffn_out = self.ffn[1](ffn_out)

        ffn_out = self.drop(ffn_out)
        # (n,b,d) -->    .view([n*b ,d]) is not needed according to the kaggle implementation
        ffn_out = self.layer_norm2(ffn_out + atn_out)  # Layer norm and Residual connection

        ## sigmoid
        ffn_out = torch.sigmoid(self.linear_out(ffn_out))
        ffn_out = ffn_out.sum(dim=-1)

        return ffn_out[:, :-1]

    def forward_update(self, interaction, question, answer):
        bs = interaction.shape[0]
        seq_len = interaction.shape[1]
        interaction = torch.where(interaction < 0, torch.zeros_like(interaction), interaction)
        question = torch.where(question < 0, torch.zeros_like(question), question)
        answer = torch.where(answer < 0, 2*torch.ones_like(answer), answer)
        device = interaction.device
        # positional embedding

        pos_index = torch.arange(self.seq_len).unsqueeze(0).to(device)

        pos_in = self.embd_pos(
            pos_index)  # making a tensor of 12 numbers, .unsqueeze(0) for converting to 2d, so as to get a 3d output

        # get the interaction embedding output

        q_emb = self.embd_q(question)

        out_in = self.embd_qa(interaction)  # (b, n) --> (b,n,d)
        #graph = torch.ones([2*self.q_num+1,2*self.q_num+1]).to(device) #graph[i,:] for the adj weight for node i

        gcn_out = self.gcn(
            x = self.embd_qa(torch.arange(2*self.q_num+1).to(device)),
            adj = self.graph
        )

        out_in = self.interaction_onehot(interaction) @ gcn_out + out_in# (bs,seq_len,|I|) * (|I|,embed_dim)


        # split the interaction embeding into v and k ( needs to verify if it is slpited or not)
        value_in = out_in
        key_in = out_in  # print('v,k ', value_in.shape)

        ## get the excercise embedding output

        query_ex = self.embd_q(question)  # (b,n) --> (b,n,d) #print(query_ex.shape)

        ## Linearly project all the embedings
        value_in = self.linear[0](value_in).permute(1, 0, 2)  # (b,n,d) --> (n,b,d)
        key_in = self.linear[1](key_in).permute(1, 0, 2)
        query_ex = self.linear[2](query_ex).permute(1, 0, 2)

        ## pass through multihead attention
        attn_mask = torch.from_numpy(
            np.triu(np.ones((self.seq_len, self.seq_len)), k=1).astype(
                'bool')).to(device)


        atn_out, _ = self.attn(query_ex, key_in, value_in
                               ,attn_mask=attn_mask
                               )
        # import matplotlib.pyplot as plt
        # print(atn_out.shape)
        # print(attn_output_weights.shape)
        # plt.imshow(attn_output_weights.detach().cpu().numpy()[0,:,:].squeeze(), cmap='plasma', origin='lower')
        # # Show the plot
        # plt.show()
        # lower triangular mask, bool, torch    (n,b,d)
        atn_out = query_ex + atn_out  # Residual connection ; added excercise embd as residual because previous ex may have imp info, suggested in paper.
        atn_out = self.layer_norm1(
            atn_out)  # Layer norm                        #print('atn',atn_out.shape) #n,b,d = atn_out.shape

        # take batch on first axis
        atn_out = atn_out.permute(1, 0, 2)  # (n,b,d) --> (b,n,d)

        ## FFN 2 layers
        # ffn_out = self.drop(
        #     self.ffn[1](
        #         nn.ReLU()
        #         (self.ffn[0](atn_out)
        #                   )
        #     )
        # )
        ffn_out = nn.ReLU()(self.ffn[0](atn_out))

        ffn_out = self.ffn[1](ffn_out)

        ffn_out = self.drop(ffn_out)
        # (n,b,d) -->    .view([n*b ,d]) is not needed according to the kaggle implementation
        ffn_out = self.layer_norm2(ffn_out + atn_out)  # Layer norm and Residual connection

        ## sigmoid
        ffn_out = torch.sigmoid(self.linear_out(ffn_out))
        ffn_out = ffn_out.sum(dim=-1)

        return ffn_out[:, :-1]

    def forward(self, interaction, question, answers):
        answers = answers.to(torch.int)
        if self.custom_feature:
            return self.forward_update(interaction, question, answers)
        else:
            return self.forward_original(interaction, question, answers)
    def input_reformat(self, q_num, s_num, question, skill, label, seq_len):
        pass

    def get_hyperparameters(self):
        hyperparameters = {
            'custom_feature' : self.custom_feature,
            'q_num': self.embd_qa.num_embeddings,
            'seq_len': self.seq_len,
            'embed_dim': self.dim,
            'heads': self.attn.num_heads,
            'dropout': self.drop.p,
        }
        return hyperparameters


class SAKT_SKILL(nn.Module):
    def __init__(self, q_num, seq_len, embed_dim, heads, dropout):
        super(SAKT_SKILL, self).__init__()
        self.seq_len = seq_len

        self.q_num = q_num
        self.qs_matrix = None
        self.dim = embed_dim
        self.heads = heads
        self.dropout = dropout

    def set_qs_matrix(self, qs_matrix):
        self.s_num = qs_matrix.shape[1]
        self.dim = self.s_num
        q_num = self.q_num
        embed_dim = self.s_num * 2
        seq_len = self.seq_len
        dropout = self.dropout
        heads = self.heads
        q_embed_weight = torch.FloatTensor(qs_matrix)  # [q_num+1,s_num]
        # Concatenate matrices as [[A, B], [B, A]]
        zero_matrix = torch.zeros_like(q_embed_weight)

        qa_embed_weight = torch.cat((torch.cat((q_embed_weight, zero_matrix), axis=1),
                                     torch.cat((zero_matrix, q_embed_weight), axis=1)), axis=0)
        q_embed_weight = torch.cat((q_embed_weight, zero_matrix), axis=1)
        print(qa_embed_weight.shape)
        print(q_embed_weight.shape)
        self.embd_qa = nn.Embedding(2 * q_num, embedding_dim=embed_dim)  # Interaction embedding
        self.embd_q = nn.Embedding(q_num, embedding_dim=embed_dim)  # Exercise embedding
        self.embd_q.weight = nn.Parameter(q_embed_weight, requires_grad=False)
        self.embd_qa.weight = nn.Parameter(qa_embed_weight, requires_grad=False)

        self.embd_pos = nn.Embedding(seq_len, embedding_dim=embed_dim)

        self.linear = nn.ModuleList(
            [nn.Linear(in_features=embed_dim, out_features=embed_dim) for x in
             range(3)])  # Linear projection for each embedding
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout)
        self.ffn = nn.ModuleList([nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True) for x in
                                  range(2)])  # feed forward layers post attention

        self.linear_out = nn.Linear(in_features=embed_dim, out_features=1, bias=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)  # output with correctnness prediction
        self.drop = nn.Dropout(dropout)

    def forward(self, input_in, input_ex, input_y):
        input_in = torch.where(input_in < 0, torch.zeros_like(input_in), input_in)
        input_ex = torch.where(input_ex < 0, torch.zeros_like(input_ex), input_ex)

        device = input_in.device

        # positional embedding
        pos_index = torch.arange(self.seq_len).unsqueeze(0).to(device)

        # making a tensor of 12 numbers, .unsqueeze(0) for converting to 2d, so as to get a 3d output
        pos_in = self.embd_pos(pos_index)

        # get the interaction embedding output
        out_in = self.embd_qa(input_in)  # (b, n) --> (b,n,d)

        out_in = out_in + pos_in

        # split the interaction embeding into v and k ( needs to verify if it is slpited or not)
        value_in = out_in
        key_in = out_in  # print('v,k ', value_in.shape)

        # get the excercise embedding output

        query_ex = self.embd_q(input_ex)  # (b,n) --> (b,n,d) #print(query_ex.shape)

        # Linearly project all the embeddings
        value_in = self.linear[0](value_in).permute(1, 0, 2)  # (b,n,d) --> (n,b,d)
        key_in = self.linear[1](key_in).permute(1, 0, 2)
        query_ex = self.linear[2](query_ex).permute(1, 0, 2)

        # pass through multihead attention
        attn_mask = torch.from_numpy(
            np.triu(np.ones((self.seq_len, self.seq_len)), k=1).astype(
                'bool')).to(device)
        atn_out, _ = self.attn(query_ex, key_in, value_in
                               #,attn_mask=attn_mask
                               )  # lower triangular mask, bool, torch    (n,b,d)
        atn_out = query_ex + atn_out  # Residual connection ; added exercise embd as residual because previous ex may have imp info, suggested in paper.
        atn_out = self.layer_norm1(
            atn_out)  # Layer norm                        #print('atn',atn_out.shape) #n,b,d = atn_out.shape

        # take batch on first axis
        atn_out = atn_out.permute(1, 0, 2)  # (n,b,d) --> (b,n,d)

        ## FFN 2 layers
        # ffn_out = self.drop(
        #     self.ffn[1](
        #         nn.ReLU()
        #         (self.ffn[0](atn_out)
        #                   )
        #     )
        # )
        ffn_out = nn.ReLU()(self.ffn[0](atn_out))

        ffn_out = self.ffn[1](ffn_out)

        ffn_out = self.drop(ffn_out)
        # (n,b,d) -->    .view([n*b ,d]) is not needed according to the kaggle implementation
        ffn_out = self.layer_norm2(ffn_out + atn_out)  # Layer norm and Residual connection

        # sigmoid
        ffn_out = torch.sigmoid(self.linear_out(ffn_out))
        ffn_out = ffn_out.sum(dim=-1)

        return ffn_out[:, :-1]

    def input_reformat(self, q_num, s_num, question, skill, label, seq_len):
        pass

    def get_hyperparameters(self):
        hyperparameters = {
            'q_num': self.embd_qa.num_embeddings,
            'seq_len': self.seq_len,
            'embed_dim': self.dim,
            'heads': self.attn.num_heads,
            'dropout': self.drop.p,
        }
        return hyperparameters


if __name__ == '__main__':
    def randomdata():
        input_in = torch.randint(0, 49, (64, 12))
        return input_in, input_in


    # Testing the model
    E = 50  # total unique exercises
    d = 128  # latent dimension
    n = 12  # sequence length

    d1, d2 = randomdata()

    print('Input shape', d1.shape)
    model = SAKT(q_num=E, seq_len=n, embed_dim=d, heads=8, dropout=0.2)
    out = model(d1, d2)
    print('Output shape', out.shape)
    print(out)
