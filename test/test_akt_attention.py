import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F





device = 'cpu'
def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    print(f"q shape {q.shape}")
    print(f"k shape {k.shape}")
    print(f"v shape {v.shape}")
    print(f"d_k : {d_k}")
    print(f"mask :{mask.shape}")
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # BS, nums_head, seqlen, seqlen
    print("Score before mask: ",scores)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    # scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)

        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    print("scores after mask",scores)
    print(v)
    output = torch.matmul(scores, v)


    return output


import unittest

class TestAttention(unittest.TestCase):

    def setUp(self) -> None:

        self.seq_len = 5
        self.embed_dim = 3
        self.bs = 1
        self.nums_head = 1
        self.d_k = self.embed_dim // self.nums_head
        self.dropout = nn.Dropout(0)
        self.gammas = nn.Parameter(torch.zeros(self.nums_head, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)


    def test_shape_correct(self):
        mask = 1 # only peek history cant peek current info
        nopeek_mask = np.triu(
            np.ones((1, 1, self.seq_len, self.seq_len)),
            k=mask
        ).astype('uint8')
        print(nopeek_mask)
        src_mask = (torch.from_numpy(nopeek_mask) == 0)

        x = torch.randn((self.bs,self.nums_head,self.seq_len,self.embed_dim))
        x = torch.tensor([[[
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]]],dtype=torch.float32)
        print(src_mask)
        print(x)
        out = attention(x,x,x,
                        d_k=self.d_k,
                        mask=src_mask,
                        dropout=self.dropout,
                        zero_pad = (mask!=0),
                        gamma = self.gammas
                        )

        print(out)
        assert out.shape == x.shape

    # def test_qas_encoder(self):
    #     y = torch.randn((self.bs, self.nums_head, self.seq_len, self.embed_dim))
    #
    #     mask = 0
    #     nopeek_mask = np.triu(
    #         np.ones((1, 1, self.seq_len, self.seq_len)),
    #         k=mask
    #     ).astype('uint8')
    #     src_mask = (torch.from_numpy(nopeek_mask) == 0)
    #
    #
    #     y = attention(mask=1, query=y, key=y, values=y)

if __name__ == '__main__':
    unittest.main()