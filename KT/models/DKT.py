import numpy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

pro_num = 10000


class DKT_AUG(nn.Module):
    def __init__(self, feature_dim, question_num, skill_num, qs_matrix, embed_dim, hidden_dim, output_dim,
                 augment_flag=False, dropout=0.2, bias=True):
        super(DKT_AUG, self).__init__()
        self.feature_dim = feature_dim
        self.question_num = question_num
        self.skill_num = skill_num
        self.qs_matrix = torch.FloatTensor(qs_matrix)  # [q_num+1,s_num+1]
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.output_dim = question_num + 1
        '''
            这个地方非常扯淡
        '''
        self.bias = bias
        self.augment_flag = augment_flag
        if augment_flag:
            self.rnn = nn.LSTM(self.embed_dim + self.skill_num + 1, hidden_dim, bias=bias, dropout=dropout,
                               batch_first=True)
        else:
            self.rnn = nn.LSTM(self.embed_dim, hidden_dim, bias=bias, dropout=dropout, batch_first=True)
        self.embedding = nn.Embedding(self.feature_dim, self.embed_dim)

        self.init_weights()
        self.linear = nn.Linear(hidden_dim, question_num + 1, bias=True)
        print(self)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, (nn.LSTM)):
                for i, weight in enumerate(m.parameters()):
                    if i < 2:
                        nn.init.orthogonal_(weight)

    def forward(self, batch_f_seq, batch_q_seq, batch_a_seq, batch_seq_len):
        '''

        :param batch_f_seq: [bs,seq_len] : 0 ~ 2*q_num+1 (0 for pad value)
        :param batch_q_seq: [bs,seq_len] : 0 ~ q_num + 1 (0 for pad value)
        :param batch_a_seq: [bs,seq_len] : 0, 1 , -1(pad_value)
        :param batch_seq_len: [bs]

        这里的最后一层的预测的方法有多种：
        1. 预测所有问题的概率，然后选取指定的相关问题
        2. 根据next_q 和当前的隐藏状态向量直接给出一个概率
        :return:
        '''

        device = batch_q_seq.device
        self.qs_matrix = self.qs_matrix.to(device)
        batch_size = batch_q_seq.shape[0]

        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(device)
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(device)

        X_one_hot_embed = torch.eye(self.feature_dim)
        Q_one_hot_embed = torch.eye(self.question_num + 1).to(device)

        # X_one_hot = F.embedding(input = batch_f_seq.cpu(),weight=X_one_hot_embed) #[bs,embed_dim]
        X_embed = self.embedding(batch_f_seq)
        Q_one_hot = F.embedding(input=batch_q_seq, weight=Q_one_hot_embed).to(device)  # [bs,question_num]
        S_ont_hot = Q_one_hot @ self.qs_matrix  # [bs ,skill_num]

        if self.augment_flag:
            input = torch.cat([X_embed, S_ont_hot], dim=-1).to(device)
        else:
            input = X_embed.to(device)
        out, (hn, cn) = self.rnn(input, (h0, c0))

        q_next = batch_q_seq[:, 1:]
        one_hot = torch.eye(int(self.question_num))
        one_hot = torch.cat((torch.zeros(1, self.question_num), one_hot), dim=0).to(device)

        next_one_hot = F.embedding(q_next, one_hot)  # select next question 1~T step

        prob_all = torch.sigmoid(self.linear(out))  # predict 1 ~ T+1
        assert torch.all(prob_all > 0)
        return self._get_next_pred(prob_all[:, :-1, :], batch_q_seq)

        prob_selected = (prob_all[:, :-1, :] * next_one_hot).sum(dim=-1)  # predict 1~T step
        print(prob_selected)  # [batch_size,seq_len]
        assert not torch.any(torch.isnan(prob_selected))
        return prob_selected

    def _get_next_pred(self, yt, questions):
        r"""
        Parameters:
            y: predicted correct probability of all questions at the next timestamp
            questions: question index matrix
        Shape:
            y: [batch_size, seq_len - 1, output_dim]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """

        device = yt.device
        one_hot = torch.eye(self.question_num + 1)

        next_qt = questions[:, 1:]

        next_qt = torch.where(next_qt != -1, next_qt, 0)  # [batch_size, seq_len - 1]

        one_hot_qt = F.embedding(next_qt.cpu(), one_hot).to(device)  # [batch_size, seq_len - 1, output_dim]

        # dot product between yt and one_hot_qt
        assert torch.all(yt != 0)
        pred = (yt * one_hot_qt).sum(dim=-1)  # [batch_size, seq_len - 1]
        assert torch.all(pred > 0)
        return pred

    def get_hyperparameters(self):
        hyperparameters = {
            'feature_dim': self.feature_dim,
            'question_num': self.question_num,
            'skill_num': self.skill_num,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'augment_flag': self.augment_flag,
            'dropout': 0.2,  # Modify this as needed
            'bias': self.bias
        }
        return hyperparameters


class DKT(nn.Module):
    def __init__(self, feature_dim, embed_dim, hidden_dim, output_dim, dropout=0.2, bias=True):
        super(DKT, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.output_dim = int(output_dim)
        self.dropout = dropout
        self.bias = bias
        self.rnn = nn.LSTM(self.embed_dim, hidden_dim, bias=bias, dropout=dropout, batch_first=True)
        self.embedding = nn.Embedding(self.feature_dim, self.embed_dim)
        self.f_out = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.init_weights()
        print(self)

    def load_pretrain_embedding(self, pretrain_file):
        import numpy as np
        q_embed = torch.FloatTensor(np.load(pretrain_file)['q_embed'])
        print(q_embed)
        print(q_embed.shape)
        assert q_embed.shape == self.embedding.weight.shape
        self.embedding.weight = q_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, (nn.LSTM)):
                for i, weight in enumerate(m.parameters()):
                    if i < 2:
                        nn.init.orthogonal_(weight)

    def _get_next_pred(self, yt, questions):
        r"""
        Parameters:
            y: predicted correct probability of all concepts at the next timestamp
            questions: question index matrix
        Shape:
            y: [batch_size, seq_len - 1, output_dim]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        device = yt.device
        one_hot = torch.eye(self.output_dim)

        # one_hot = torch.cat((torch.zeros(1, self.output_dim), one_hot), dim=0)

        next_qt = questions[:, 1:]

        next_qt = torch.where(next_qt != -1, next_qt, 0)  # [batch_size, seq_len - 1]

        one_hot_qt = F.embedding(next_qt.cpu(), one_hot).to(device)  # [batch_size, seq_len - 1, output_dim]

        # dot product between yt and one_hot_qt
        # assert torch.all(one_hot_qt.sum(dim=-1)>0)
        pred = (yt * one_hot_qt).sum(dim=-1)  # [batch_size, seq_len - 1]
        # assert torch.all(pred != 0)
        return pred

    def inference(self, X: torch.LongTensor, labels: torch.LongTensor, q_next):
        '''

        :param X: [seq_len]
        :param y: [seq_len]
        :param q_next: scalar
        :return:
        '''
        device = X.device
        batch_size = X.shape[0]

        mask = (labels <= 0).cpu()

        amp = torch.ones_like(labels.cpu().int()) * pro_num

        amp = amp.masked_fill(mask, 1)

        tmp = torch.where(labels < 0, torch.zeros_like(labels), labels)
        features = (X.cpu() + torch.mul(tmp.cpu().int(), amp)).to(device)
        features = torch.IntTensor(features.cpu().numpy()).to(device)

        assert torch.min(features) > -1
        assert torch.max(features) < self.feature_dim

        features = self.embedding(features).to(device)
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(features.device)
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(features.device)

        out, _ = self.rnn(features, (h0, c0))

        prob_all = torch.sigmoid(self.f_out(out))
        return prob_all[:, -1, q_next]

    def forward_with_skill(self, features, questions, skills, labels, q_matrix):
        device = questions.device
        qm_tmp = q_matrix.to(device)
        skill_one_hot = torch.nn.functional.embedding(questions, qm_tmp)

        batch_size = questions.shape[0]

        mask = (labels <= 0).cpu()

        amp = torch.ones_like(labels.cpu().int()) * pro_num

        amp = amp.masked_fill(mask, 1)

        tmp = torch.where(labels < 0, torch.zeros_like(labels), labels)
        features = (questions.cpu() + torch.mul(tmp.cpu().int(), amp)).to(device)
        features = torch.IntTensor(features.cpu().numpy()).to(device)

        assert torch.min(features) > -1
        assert torch.max(features) < self.feature_dim

        features = self.embedding(features).to(device)

        features = torch.cat([features, skill_one_hot], dim=-1)
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(features.device)
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(features.device)

        out, _ = self.rnn(features, (h0, c0))
        yt = self.f_out(out)
        yt = torch.sigmoid(yt)

        yt = yt[:, :-1, :]
        return self._get_next_pred(yt, questions)

    def forward_without_skill(self, features, questions, labels):
        device = features.device

        batch_size = features.shape[0]

        mask = (labels <= 0).cpu()

        amp = torch.ones_like(labels.cpu().int()) * pro_num

        amp = amp.masked_fill(mask, 1)

        features = torch.IntTensor(features.cpu().numpy()).to(device)

        # assert torch.min(features) > -1
        #
        # assert torch.max(features) < self.feature_dim

        features = self.embedding(features).to(device)

        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(features.device)
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(features.device)

        out, _ = self.rnn(features, (h0, c0))
        yt = self.f_out(out)
        yt = torch.sigmoid(yt)

        yt = yt[:, :-1, :]
        # def assert_non_zero(t):
        #     assert torch.all(t > torch.zeros_like(t))
        # assert_non_zero(yt)
        # assert_non_zero(self._get_next_pred(yt, questions))
        return self._get_next_pred(yt, questions)

    def forward(self, batch):
        features = batch['question_answer']
        questions = batch['question']
        skills = batch['skill']
        labels = batch['answer']
        seq_len = batch['seq_len']
        return self.forward_without_skill(features, questions, labels)

    def load(self, ):
        pass

    def get_hyperparameters(self):
        hyperparameters = {
            'feature_dim': self.feature_dim,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'bias': self.bias
        }
        return hyperparameters


class DKT_PLUS(nn.Module):
    def __init__(self, s_num: int, q_num: int, hidden_dim: int, output_dim: int, embed_dim: int, dropout: float = 0.8,
                 bias: bool = True):
        super(DKT_PLUS, self).__init__()

        # Your initialization code here

        self.s_num = s_num
        self.q_num = q_num
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.feature_type = ['s-multihot', 'q-embed', 'q-onehot', 's-embed'][0]
        # Example: Define layers using the provided parameters
        self.embedding = nn.Embedding(q_num, embed_dim)
        input_dim = embed_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def set_qs_matrix(self, qs_matrix):
        self.qs_matrix = torch.FloatTensor(qs_matrix)  # [q_num+1,s_num]
        # Concatenate matrices as [[A, B], [B, A]]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, (nn.LSTM)):
                for i, weight in enumerate(m.parameters()):
                    if i < 2:
                        nn.init.orthogonal_(weight)

    def forward(self, interaction, question, answer):
        device = interaction.device

        batch_size = interaction.shape[0]

        mask = (answer <= 0).cpu()

        amp = torch.ones_like(answer.cpu().int()) * self.s_num

        amp = amp.masked_fill(mask, 1)

        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(feature.device)
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(feature.device)

        out, _ = self.rnn(feature, (h0, c0))
        yt = self.f_out(out)
        yt = torch.sigmoid(yt)

        yt = yt[:, :-1, :]

        def assert_non_zero(t):
            assert torch.all(t > torch.zeros_like(t))

        assert_non_zero(yt)
        assert_non_zero(self._get_next_pred(yt, question))
        return self._get_next_pred(yt, question)


class DKT_PEBG(torch.nn.Module):
    def __init__(self, pro_num, skill_num, hidden_dim, output_dim, embed_dim, p_embed_weight=None):
        super(DKT_PEBG, self).__init__()
        self.pro_num = pro_num
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pro_embed = nn.Embedding(pro_num, self.embed_dim)
        self.pro_embed.weight.requires_grad = False

        self.gru = nn.GRU(input_size=2 * embed_dim, hidden_size=hidden_dim, batch_first=True, dropout=0.2)
        self.lstm = nn.LSTM(input_size=2 * embed_dim, hidden_size=hidden_dim, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, pro_num, bias=True)
        self.output_dim = pro_num

    def load_pretrain_embedding(self, pretrain_file):
        import numpy as np
        q_embed = torch.FloatTensor(np.load(pretrain_file)['q_embed'])
        '''
        在预训练embedding的时候，我们没有考虑问题0对吧
        0在输入序列中是用来填充的
        '''

        q_embed = torch.concat([torch.zeros([1, q_embed.shape[1]]), q_embed])
        print(q_embed.shape)
        print(self.pro_embed.weight.shape)
        if q_embed is not None:
            assert q_embed.shape == self.pro_embed.weight.shape
            self.pro_embed = nn.Embedding.from_pretrained(q_embed)
            self.pro_embed.weight.requires_grad = False

    def inference(self, X: torch.LongTensor, y: torch.LongTensor, q_next):
        '''

        :param X: [seq_len]
        :param y: [seq_len]
        :param q_next: scalar
        :return:
        '''
        assert X.shape == y.shape
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(X.device)
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(X.device)
        embed_X = self.pro_embed(X)  # [batch_size,seq_len,embed_dim]
        embed_X = torch.concat([embed_X, embed_X], dim=2)

        one_zero = torch.concat([torch.ones(self.embed_dim), torch.zeros(self.embed_dim)]).unsqueeze(dim=0)
        zero_one = torch.concat([torch.zeros(self.embed_dim), torch.ones(self.embed_dim)]).unsqueeze(dim=0)

        embedding_y = torch.concat([one_zero, zero_one, torch.zeros_like(one_zero)], dim=0).to(X.device)

        y_ = y.clone().int()
        y_[y_ == -1] = 2

        mask_y = F.embedding(input=y_, weight=embedding_y, padding_idx=2)

        embed_X = torch.mul(embed_X, mask_y)

        # print(embed_X.shape)
        # print(h0.shape)
        out, (hn, cn) = self.lstm(embed_X, (h0, c0))
        # out,hn = self.gru(embed_X,h0)
        # print(self.pro_num)

        one_hot_matrix = torch.eye(int(self.pro_num)).to(X.device)

        one_hot = F.embedding(X[:, 1:], one_hot_matrix)  # select next question 1~T step
        # print(one_hot.shape)

        prob_all = torch.sigmoid(self.linear(out))
        return prob_all[:, -1, q_next]

    def forward(self, X: torch.Tensor, y: torch.Tensor):
        '''

        :param X: [batch_size,seq_len]
        :param y: [batch_size,seq_len]
        :return:
        '''
        assert X.shape == y.shape
        assert torch.max(X) < self.pro_embed.num_embeddings
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(X.device)
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim)).to(X.device)
        embed_X = self.pro_embed(X)  # [batch_size,seq_len,embed_dim]
        embed_X = torch.concat([embed_X, embed_X], dim=2)

        one_zero = torch.concat([torch.ones(self.embed_dim), torch.zeros(self.embed_dim)]).unsqueeze(dim=0)
        zero_one = torch.concat([torch.zeros(self.embed_dim), torch.ones(self.embed_dim)]).unsqueeze(dim=0)

        embedding_y = torch.concat([one_zero, zero_one, torch.zeros_like(one_zero)], dim=0).to(X.device)

        y_ = y.clone().int()
        y_[y_ == -1] = 2

        mask_y = F.embedding(input=y_, weight=embedding_y, padding_idx=2)

        embed_X = torch.mul(embed_X, mask_y)

        # print(embed_X.shape)
        # print(h0.shape)
        out, (hn, cn) = self.lstm(embed_X, (h0, c0))

        # out,hn = self.gru(embed_X,h0)
        # print(self.pro_num)

        one_hot = torch.eye(int(self.pro_num)).to(X.device)
        one_hot = torch.cat((torch.zeros(1, self.pro_num, device=X.device), one_hot), dim=0)
        next_one_hot = F.embedding(X[:, 1:], one_hot)  # select next question 1~T step
        # print(one_hot.shape)

        prob_all = torch.sigmoid(self.linear(out))  # predict 1 ~ T+1

        prob_selected = (prob_all[:, :-1, :] * next_one_hot).sum(dim=-1)  # predict 1~T step

        return prob_selected

    def get_hyperparameters(self):
        hyperparameters = {
            'pro_num': self.pro_num,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dropout': 0.2  # Modify this as needed
        }
        return hyperparameters


if __name__ == '__main__':

    device = 'cuda'
    hidden_dim = 100
    pro_num = 10000

    skill_num = 100
    qs_matrix = np.zeros([pro_num, skill_num])

    feature_dim = 100
    embed_dim = 100
    output_dim = 100
    model = DKT_AUG(
        feature_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        qs_matrix,
        skill_num,
        dropout=0.2,
        bias=True
    )


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
