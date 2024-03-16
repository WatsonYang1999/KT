import argparse
def set_parser():
    def str_to_bool(s):
        return s.lower() == 'true'

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
    parser.add_argument('--model', type=str, default='DKVMN_RE',
                        help='Model type to use, support GKT,SAKT,QGKT and DKT.')
    '''
    Available Dataset:
    ednet
    beihang
    assist09-q
    assist09-s
    assist17-s
    '''

    parser.add_argument('--dataset', type=str, default='ednet-re', help='Dataset You Wish To Load')
    # parser.add_argument('--dataset', type=str, default='ednet_qs', help='Dataset You Wish To Load')

    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Model Parameters Directory')
    parser.add_argument('--train_from_scratch', type=str_to_bool, default=False,
                        help='If you need to retrain the model from scratch')

    parser.add_argument('--eval', type=str_to_bool, default='False',
                        help='Evaluate model to find some interesting insights')
    parser.add_argument('--custom_data', type=str_to_bool, default='False',
                        help='Use your own custom data')

    parser.add_argument('--skill_level_eval', type=str_to_bool, default='False',
                        help='Evaluate the model on skill_level and try to address the label leakage issue')

    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--current_epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=20, help='Total Epochs.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=200)
    parser.add_argument('--shuffle', type=str_to_bool, default='False')
    parser.add_argument('--cuda', type=str_to_bool, default='True')
    # some model hyper-parameters

    parser.add_argument('--hidden_dim', type=int, default=50, help='')
    parser.add_argument('--embed_dim', type=int, default=128, help='')
    parser.add_argument('--output_dim', type=int, default=100, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')
    parser.add_argument('--memory_size', type=int, default=20, help='')
    parser.add_argument('--n_heads', type=int, default=4, help='number of multi-attention heads')

    # graph related paramaters
    parser.add_argument('--edge_types', type=int, default=2, help='')
    parser.add_argument('--graph_type', type=str, default='PAM', help='')
    parser.add_argument('--device', type=str, default='cuda', help='')

    parser.add_argument('--s_num', type=int, default=-1, help='')
    parser.add_argument('--q_num', type=int, default=-1, help='')

    parser.add_argument('--data_augment', type=str_to_bool, default='False', help='')
    parser.add_argument('--pretrain', type=str, default='load', help='scratch or load or no')
    parser.add_argument('--pretrain_embed_file', type=str, default='', help='path of the pretrain weight file')

    parser.add_argument('--log_file', type=str, default='', help='path of the logging file')
    return parser


