import logging
import os
from datetime import datetime

import torch

from KT.models.Loss import KTLoss
from KT.models.Pretrain import embedding_pretrain
from KT.train import train, evaluate
from KT.util.args import set_parser
from KT.util.kt_util import load_model, load_dataset, reformat_datatime, get_model_size

parser = set_parser()
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)
if args.cuda:
    assert torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
args.device = device

args.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')

# load dataset
data_loaders = load_dataset(args)

train_loader = data_loaders['train_loader']
test_loader = data_loaders['test_loader']
args.qs_matrix = data_loaders['qs_matrix']

args.metrics = ['train_auc', 'train_loss', 'train_acc', 'train_rmse','val_auc', 'val_loss', 'val_acc','val_rmse']
# print(train_loader.dataset.get_question_trans_graph())

if args.pretrain == 'scratch':
    embedding_pretrain(args, train_loader.dataset, test_loader.dataset, args.qs_matrix)
elif args.pretrain == 'load':
    if args.pretrain_embed_file == '':
        args.pretrain_embed_file = 'Dataset/' + args.dataset + '/embed_pretrain.npz'
    else:
        args.pretrain_embed_file = 'Dataset/' + args.dataset + '/' + args.pretrain_embed_file

# load model
model, optimizer = load_model(args)

model = model.to(args.device)
get_model_size(model)
kt_loss = KTLoss()


def set_logger(args):
    time_program_begin = datetime.now()
    log_file_name = '-'.join([args.dataset.__str__(),
                              args.model.__str__(),
                              reformat_datatime(time_program_begin)]) + '.txt'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output logs to the console
            logging.FileHandler(os.path.join('Log', log_file_name))
        ]
    )
    logging.info(args.__str__())
    logging.info('-----------------------------------------------------------')


set_logger(args)

if not args.eval:
    # train model
    time_training_begin = datetime.now()
    logs = train(model, data_loaders, optimizer, kt_loss, args)
    time_training_end = datetime.now()
    config = args


    # config the log content here
    def log_train():
        best_epoch = -1
        best_val_auc = -1
        metrics = args.metrics
        metric_select = 'val_auc'
        greater_is_better = 1
        for idx, metric_i in enumerate(logs[metric_select]):
            if (metric_i - best_val_auc) * greater_is_better > 0:
                best_val_auc = metric_i
                best_epoch = idx
        for _ in metrics:
            output = f"Best {metric_select} in epoch {best_epoch}: "
            for m in metrics:
                output += m + ":  " + f"{logs[m][best_epoch]}  "
            logging.info(output)

        logging.info('-----------------------------------------------------------')
        logging.info(str(logs))

        delta_time = datetime.timestamp(time_training_end) - datetime.timestamp(time_training_end)
        logging.info('Delta Time : ' + delta_time.__str__())
        logging.info('\n')


    log_train()

else:
    logging.info("---------------------------evaluating---------------------------------")
    custom_data_loaders = {'custom_dataset': test_loader}
    evaluate(
        model=model,
        data_loaders=custom_data_loaders,
        loss_func=kt_loss,
        cuda=args.cuda
    )
    # for data_loader in data_loaders.values():
    #     metric_list = rank_data_performance(model=model,
    #                                         data_loader=data_loader,
    #                                         loss_func=kt_loss,
    #                                         cuda=args.cuda)
    #
    #     for metric, seq_list in metric_list.items():
    #         print(f"Rank the dataset sequences by {metric}")
    #         for seq in seq_list:
    #             pass

# save training log, including essential config: dataset , model hyperparameters, best metrics

# test model

# test(model, train_loader, optimizer, kt_loss, args.n_epochs, args.cuda)


if __name__ == '__main__':
    pass
