import torch
import os
from util.kt_util import load_model, load_dataset
from util.args import set_parser
from KT.train import train
from KT.models.Loss import KTSequenceLoss
from KT.models.Pretrain import embedding_pretrain
from datetime import datetime
import logging

time_program_begin = datetime.now()

parser = set_parser()
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.checkpoint_dir = os.path.join(os.getcwd(),'Checkpoints')

# load dataset
data_loaders = load_dataset(args)

def reformat_datatime(dt: datetime):
    formatted_time = dt.strftime("%y-%m-%d_%H-%M-%S")
    return formatted_time


log_file_name = '-'.join([args.dataset.__str__(),
                          args.model.__str__(),
                          reformat_datatime(time_program_begin)]) + '.txt'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output logs to the console
        logging.FileHandler(os.path.join('Log', log_file_name))
    ]
)
logging.info(args.__str__())

train_loader = data_loaders['train_loader']
test_loader = data_loaders['test_loader']
args.qs_matrix = data_loaders['qs_matrix']

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
kt_loss = KTSequenceLoss()

# train model
time_training_begin = datetime.now()
logs = train(model, data_loaders, optimizer, kt_loss, args)
time_training_end = datetime.now()
config = args

log_file_name = '-'.join([args.dataset.__str__(),
                          args.model.__str__(),
                          reformat_datatime(time_program_begin)]) + '.txt'

# config the log content here
logging.info(args.__str__())
logging.info('-----------------------------------------------------------')
delta_time = datetime.timestamp(time_training_end) - datetime.timestamp(time_training_end)
logging.info('Delta Time : ' + delta_time.__str__())
logging.info('\n')
best_epoch = -1
best_val_auc = -1
metrics = ['train_auc', 'train_loss', 'train_acc', 'val_auc', 'val_loss', 'val_acc']
metric_select = 'val_auc'
greater_is_better = 1
for idx, metric_i in enumerate(logs[metric_select]):
    if (best_val_auc - metric_i) * greater_is_better > 0:
        best_val_auc = metric_i
        best_epoch = idx
for m in metrics:
    output = f"Best {metric_select} in epoch {best_epoch}: "
    for m in metrics:
        output += m + ":  " + f"{logs[m][best_epoch]}  "
    logging.info(output)

logging.info('-----------------------------------------------------------')
logging.info(str(logs))

# save training log, including essential config: dataset , model hyperparameters, best metrics

# test model

# test(model, train_loader, optimizer, kt_loss, args.n_epochs, args.cuda)


if __name__=='__main__':
    pass

