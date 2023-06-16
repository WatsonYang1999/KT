import argparse
import torch
from KT.utils import load_model, load_dataset
from KT.train import train, test
from KT.models.Loss import KTLoss
from KT.models.Pretrain import embedding_pretrain

def set_parser():
    parser = argparse.ArgumentParser()
    '''
    Available Models
    DKT
    GKT
    DKVMN
    AKT
    SAKT
    SAINT
    '''
    parser.add_argument('--model', type=str, default='DKT_AUG', help='Model type to use, support GKT,SAKT,QGKT and DKT.')
    '''
    Available Dataset:
    ednet
    beihang
    assist09-q
    assist09-s
    assist17-s
    '''
    #parser.add_argument('--dataset', type=str, default='assist09-q', help='Dataset You Wish To Load')
    parser.add_argument('--dataset', type=str, default='ednet_qs', help='Dataset You Wish To Load')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Model Parameters Directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--n_epochs', type=int, default=200, help='Total Epochs.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=True)
    # some model hyper-parameters

    parser.add_argument('--hidden_dim', type=int, default=50, help='')
    parser.add_argument('--embed_dim', type=int, default=128, help='')
    parser.add_argument('--output_dim', type=int, default=100, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')
    parser.add_argument('--memory_size', type=int, default=20, help='')
    # graph related paramaters
    parser.add_argument('--edge_types', type=int, default=2, help='')
    parser.add_argument('--graph_type', type=str, default='Dense', help='')
    parser.add_argument('--device', type=str, default='cuda', help='')

    parser.add_argument('--s_num', type=int, default=-1, help='')
    parser.add_argument('--q_num', type=int, default=-1, help='')

    parser.add_argument('--data_augment',type=bool,default=False,help='')
    parser.add_argument('--pretrain',type=str,default='load',help='scratch or load or no')
    parser.add_argument('--pretrain_embed_file',type=str,default='',help='path of the pretrain weight file')

    return parser

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = set_parser()
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

# load dataset

data_loaders = load_dataset(args)

print(args)

train_loader = data_loaders['train_loader']
test_loader = data_loaders['test_loader']
args.qs_matrix = data_loaders['qs_matrix']
print(args.qs_matrix)

if args.pretrain == 'scratch':
    embedding_pretrain(args,train_loader.dataset,test_loader.dataset,args.qs_matrix)
elif args.pretrain == 'load':
    if args.pretrain_embed_file == '':
        args.pretrain_embed_file = 'Dataset/'+args.dataset + '/embed_pretrain.npz'
    else:
        args.pretrain_embed_file = 'Dataset/' + args.dataset+'/'+args.pretrain_embed_file
# load model
model, optimizer = load_model(args)
model = model.to(args.device)
kt_loss = KTLoss()

# train model
train_logs = train(model, data_loaders, optimizer, kt_loss, args.n_epochs, args.cuda)

config = args

# save training log, including essential config: dataset , model hyperparameters, best metrics


# test model

test(model, train_loader, optimizer, kt_loss, args.n_epochs, args.cuda)
