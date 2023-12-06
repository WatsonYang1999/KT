import torch
from torch import nn
from sklearn.metrics import roc_auc_score


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels.reshape(-1)).double()
    correct = correct.sum()
    return correct / len(labels)


class KTLoss(nn.Module):

    def __init__(self):
        super(KTLoss, self).__init__()

    def forward(self, pred_answers, real_answers):
        r"""
        Parameters:
            pred_answers: the correct probability of questions answered at the next timestamp
            real_answers: the real results(0 or 1) of questions answered at the next timestamp
        Shape:
            pred_answers: [batch_size, seq_len - 1]
            real_answers: [batch_size, seq_len]
        Return:
        """
        assert pred_answers.shape[1] == real_answers.shape[1] - 1

        real_answers = real_answers[:, 1:]  # timestamp=1 ~ T
        # real_answers shape: [batch_size, seq_len - 1]
        # Here we can directly use nn.BCELoss, but this loss doesn't have ignore_index function
        answer_mask = torch.ne(real_answers, -1)

        pred_one, pred_zero = pred_answers, 1.0 - pred_answers  # [batch_size, seq_len - 1]
        assert not torch.any(torch.isnan(pred_answers))
        # calculate auc and accuracy metrics
        try:
            y_true = real_answers[answer_mask].cpu().detach().numpy()
            y_pred = pred_one[answer_mask].cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred)  # may raise ValueError
            output = torch.cat((pred_zero[answer_mask].reshape(-1, 1), pred_one[answer_mask].reshape(-1, 1)), dim=1)
            label = real_answers[answer_mask].reshape(-1, 1)
            acc = accuracy(output, label)
            acc = float(acc.cpu().detach().numpy())
        except ValueError as e:
            auc, acc = -1, -1

        # calculate NLL loss
        '''
        Log 不能有0，这他妈为啥会出现有0的呢
        '''

        pred_one[answer_mask] = torch.log(pred_one[answer_mask])

        pred_zero[answer_mask] = torch.log(pred_zero[answer_mask])

        pred_answers = torch.cat((pred_zero.unsqueeze(dim=1), pred_one.unsqueeze(dim=1)), dim=1)

        # pred_answers shape: [batch_size, 2, seq_len - 1]
        nll_loss = nn.NLLLoss(ignore_index=-1)  # ignore masked values in real_answers

        loss = nll_loss(pred_answers, real_answers.long())
        if torch.isnan(loss):
            raise ValueError(f"Nan value in loss:\n pred: {pred_answers} \nlabels: {real_answers}")
        return loss, auc, acc


if __name__ == '__main__':
    loss = KTLoss()

    pred = torch.FloatTensor([[0.7922, 0.8074, 0.7945, 0.7944, 0.7269, 0.7651, 0.7109, 0.7056, 0.7413,
                               0.7427, 0.7115, 0.7265, 0.7058, 0.7309, 0.6964, 0.6870, 0.7407, 0.6885,
                               0.6827, 0.6888, 0.7002, 0.6729, 0.5825, 0.6657, 0.7178, 0.5950, 0.6010,
                               0.6375, 0.5724, 0.6385, 0.6420, 0.5379, 0.6297, 0.5870, 0.6199, 0.6406,
                               0.6412, 0.6571, 0.5537, 0.6379, 0.6146, 0.6308, 0.5895, 0.6913, 0.6533,
                               0.6823, 0.6006, 0.6656, 0.6085, 0.6436, 0.7023, 0.6911, 0.6589, 0.6342,
                               0.6904, 0.6144, 0.6008, 0.6273, 0.6363, 0.6747, 0.6159, 0.6290, 0.5616,
                               0.5601, 0.6045, 0.6287, 0.5835, 0.5980, 0.5899, 0.5685, 0.6215, 0.5391,
                               0.6340, 0.5964, 0.5470, 0.5710, 0.5823, 0.6278, 0.6123, 0.5801, 0.5394,
                               0.5165, 0.5181, 0.5210, 0.5589, 0.5870, 0.6378, 0.5599, 0.5367, 0.6440,
                               0.5477, 0.6081, 0.5674, 0.6097, 0.4649, 0.6085, 0.5437, 0.6081, 0.6005,
                               0.6043, 0.5895, 0.5880, 0.5655, 0.5637, 0.5929, 0.5490, 0.5376, 0.5662,
                               0.5455, 0.5463, 0.5519, 0.5757, 0.5440, 0.5497, 0.5050, 0.5668, 0.4522,
                               0.5586, 0.4791, 0.5583, 0.5696, 0.5418, 0.5776, 0.5508, 0.5468, 0.5388,
                               0.5259, 0.5246, 0.5594, 0.5316, 0.5071, 0.5762, 0.4833, 0.6006, 0.5087,
                               0.5434, 0.4925, 0.5694, 0.5859, 0.4735, 0.4979, 0.5128, 0.5204, 0.4777,
                               0.5110, 0.5045, 0.5446, 0.5074, 0.5798, 0.5714, 0.4871, 0.5140, 0.5218,
                               0.5801, 0.5438, 0.5261, 0.5102, 0.5771, 0.4821, 0.4694, 0.5543, 0.4911,
                               0.4460, 0.5323, 0.5437, 0.4536, 0.4370, 0.5575, 0.4809, 0.4620, 0.5184,
                               0.4892, 0.5583, 0.4935, 0.5092, 0.5032, 0.4398, 0.5458, 0.4911, 0.4649,
                               0.5027, 0.5220, 0.5206, 0.4876, 0.5067, 0.4723, 0.5561, 0.5400, 0.4513,
                               0.4609, 0.5199, 0.4871, 0.4864, 0.5477, 0.4512, 0.4882, 0.5235, 0.5526,
                               0.5386]])
    labels = torch.FloatTensor([[1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                                 -1., -1., -1., -1.]])

    print(loss(pred, labels))
