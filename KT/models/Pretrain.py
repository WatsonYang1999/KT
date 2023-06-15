import math

import torch

import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np
class PretrainModel(torch.nn.Module):
    def __init__(self,q_num,s_num,embed_dim):
        super(PretrainModel, self).__init__()
        #如何初始化这个embedding matrix不知道有没有讲究
        def truncated_normal_(tensor, mean=0., std=1.):
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor
        s_embed_init = torch.randn([s_num,embed_dim])

        p_embed_init = torch.randn([q_num,embed_dim])

        norm = True
        if norm :
            s_embed_init = truncated_normal_(s_embed_init, std=0.1)
            p_embed_init = truncated_normal_(p_embed_init,std=0.1)
            # d_embed_init = truncated_normal_(d_embed_init,std=0.1)
        # init_data = np.load('init.npz')
        # s_embed_init = torch.FloatTensor(init_data['skill_embedding_init'])
        # p_embed_init = torch.FloatTensor(init_data['pro_embedding_init'])

        self.skill_embedding_matrix = Variable(s_embed_init,requires_grad=True)

        self.pro_embedding_matrix = Variable(p_embed_init,requires_grad=True)
        # self.diff_embedding_matrix = Variable(d_embed_init,requires_grad=True)




    def forward(self,question):

        # one_hot_pro = torch.eye(pro_num)
        # pro_one_hot = F.embedding(input=pro,weight=one_hot_pro) #[]
        # pro_embed = torch.matmul(pro_one_hot,self.pro_embedding_matrix)
        q_embed = F.embedding(input=question,weight=self.pro_embedding_matrix)

        # pro-pro

        qq_logits = q_embed @ self.pro_embedding_matrix.T #[bs,embed_dim]  [embed_dim,q_num]

        qq_logits = torch.reshape(qq_logits,[-1])
        # pro-skill

        qs_logits = torch.reshape(q_embed @ self.skill_embedding_matrix.T,[-1]) #[bs, embed_dim] [ embed_dim]

        ss_logits = torch.reshape(self.skill_embedding_matrix @ self.skill_embedding_matrix.T, [-1])



        # feature fuse
        # skill_embed = torch.matmul(pro_skill_targets, self.skill_embedding_matrix) / torch.sum(pro_skill_targets)
        # pro_final_embed, p = pnn1([pro_embed, skill_embed, diff_feat_embed])
        # mse = torch.mean(torch.square(p - auxiliary_targets))

        #loss = mse + cross_entropy_pro_skill + cross_entropy_pro_pro + cross_entropy_skill_skill
        #loss = cross_entropy_pro_skill + cross_entropy_pro_pro + cross_entropy_skill_skill

        return qq_logits,qs_logits,ss_logits

class PretrainLoss(torch.nn.Module):
    def __init__(self):
        super(PretrainLoss, self).__init__()


    def forward(self,qq_logits,qq_target,qs_logits,qs_target,ss_logits,ss_target):
        def sigmoid_cross_entropy_with_logits(logits,labels):
            import torch.nn.functional as F
            probs = F.sigmoid(logits)
            bce_loss = F.binary_cross_entropy_with_logits(probs, labels)
            return bce_loss

        qq_loss = sigmoid_cross_entropy_with_logits(qq_logits,qq_target)
        qs_loss = sigmoid_cross_entropy_with_logits(qs_logits,qs_target)
        ss_loss = sigmoid_cross_entropy_with_logits(ss_logits,ss_target)
        return qq_loss+qs_loss+ss_loss



def embedding_pretrain(args,train_set,test_set,qs_matrix):
    qs_matrix = qs_matrix[1:,:]

    q_num = args.q_num
    s_num = args.s_num
    embed_dim = args.embed_dim

    # qs_matrix = torch.FloatTensor([
    #     [1,0,0],
    #     [0,1,0],
    #     [1,0,0]
    # ])
    def qq_similarity(qs_matrix):

        qq = qs_matrix@qs_matrix.T

        def mag(v):

            return math.sqrt(sum(pow(element, 2) for element in v))

        def try_mag():
            print(mag(torch.FloatTensor([3.,4.])))

        # try_mag()
        magnitude = torch.FloatTensor([mag(qs_matrix[i,:]) for i in range(0,qs_matrix.shape[0])])

        magnitude = magnitude.unsqueeze(-1)


        mag_matrix = magnitude @ magnitude.T

        # assert not torch.any(mag_matrix==0)
        assert mag_matrix.shape == qq.shape

        qq = qq/mag_matrix
        qq = torch.nan_to_num(qq,nan=0)
        return qq
    def ss_similarity(sq_matrix):
        return qq_similarity(sq_matrix)

    qq_dense = qq_similarity(qs_matrix=qs_matrix)
    ss_dense = ss_similarity(sq_matrix=qs_matrix.T)

    torch.set_printoptions(precision=2)
    def draw_heat_map(matrix):
        import matplotlib.pyplot as plt
        print(matrix[0,0])
        plt.imshow(matrix, cmap="YlGnBu", interpolation='nearest')
        plt.show()

    # draw_heat_map(ss_dense.numpy())
    qs_dense = qs_matrix





    hidden_dim = 128  # hidden dim in PNN
    keep_prob = 0.5
    lr = 0.01
    bs = 64
    epochs = 200
    model_flag = 0

    model = PretrainModel(q_num,s_num,embed_dim)

    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.BCEWithLogitsLoss()
    criterion3 = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([model.skill_embedding_matrix, model.pro_embedding_matrix], lr=lr)
    train_steps = int(math.ceil(q_num / float(bs)))
    for i in range(epochs):
        model.train()
        loss_list = []

        for m in range(train_steps):
            optimizer.zero_grad()
            b, e = m * bs, min((m + 1) * bs, q_num)
            questions = torch.IntTensor(np.arange(b, e).astype(np.int32))
            batch_qs_targets = torch.FloatTensor(qs_dense[b:e, :])

            batch_qq_targets = torch.FloatTensor(qq_dense[b:e, :])


            qq_logits, qs_logits, ss_logits = model(questions)
            # batch_loss = criterion(qq_logits,
            #                        batch_qq_targets.reshape([-1]),
            #                        qs_logits,
            #                        batch_qs_targets.reshape([-1]),
            #                        ss_logits,
            #                        skill_skill_dense.reshape([-1]))


            batch_loss = criterion1(qq_logits, batch_qq_targets.reshape([-1])) \
                         + criterion2(qs_logits, batch_qs_targets.reshape([-1])) \
                         + criterion3(ss_logits, ss_dense.reshape([-1]))
            batch_loss.backward()
            optimizer.step()
            loss_list.append(batch_loss.cpu().detach().numpy())

        print(np.mean(loss_list))

    np.savez('Dataset/'+args.dataset+'/embed_pretrain.npz',
             q_embed = model.pro_embedding_matrix.detach().numpy(),
             s_embed = model.skill_embedding_matrix.detach().numpy())
    exit(-1)
    return model.skill_embedding_matrix,model.pro_embedding_matrix
