import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score,mean_squared_error


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels.reshape(-1)).double()
    correct = correct.sum()
    return correct / len(labels)



def calculate_rmse_sklearn(actual_values, predicted_values):
    # Ensure both arrays have the same length
    if len(actual_values) != len(predicted_values):
        raise ValueError("Input arrays must have the same length")

    # Convert input lists to NumPy arrays
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)

    # Calculate mean squared error
    mean_squared_error_value = mean_squared_error(actual_values, predicted_values)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error_value)

    return rmse


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
            rmse = calculate_rmse_sklearn(y_true, y_pred)
            output = torch.cat((pred_zero[answer_mask].reshape(-1, 1), pred_one[answer_mask].reshape(-1, 1)), dim=1)
            label = real_answers[answer_mask].reshape(-1, 1)
            acc = accuracy(output, label)
            acc = float(acc.cpu().detach().numpy())


        except ValueError as e:

            auc, acc,rmse = -1, -1,-1

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
        return loss, auc, acc,rmse


if __name__ == '__main__':
    loss = KTLoss()

    pred = torch.FloatTensor([[0.01,0.01,0.5,0.99]])
    labels = torch.IntTensor([[1., 0.,1.,1.,1.]])

    print(loss(pred, labels))
