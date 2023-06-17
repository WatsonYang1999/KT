import json

import numpy as np
import torch
import os
from KT.KTDataloader import KTDataset
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, Subset


# divide train test set
def train_test_split(data, split=0.8):
    n_samples = data[0].shape[0]
    indices = np.random.permutation(n_samples)
    split_point = int(n_samples * split)
    train_data, test_data = [], []
    for d in data:
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]
        train_data.append(d[train_idx])
        test_data.append(d[test_idx])
    return train_data, test_data


def load_assist09_q(args):
    data_dir = 'Dataset\\' + args.dataset
    data = np.load(os.path.join(data_dir, args.dataset + '.npz'))
    y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']

    args.s_num, args.q_num = data['skill_num'], data['problem_num']

    train_data, test_data = train_test_split([y, skill, problem, real_len])  # [y, skill, pro, real_len]
    train_y, train_skill, train_problem, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
    test_y, test_skill, test_problem, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
    args.q_num = max(np.max(train_problem), np.max(test_problem))
    train_set = KTDataset(args.q_num, args.s_num, train_problem, train_skill, train_y, train_real_len,
                          max_seq_len=args.max_seq_len)
    test_set = KTDataset(args.q_num, args.s_num, test_problem, test_skill, test_y, test_real_len,
                         max_seq_len=args.max_seq_len)
    if args.data_augment:
        print('Use Data Augmentation')
        train_set.augment()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=args.shuffle)
    '''
        Modification Required
        need a better way to load qs_matrix cuz the original file is like a crap of shit 
    '''
    return {'train_loader': train_loader, 'test_loader': test_loader, 'qs_matrix': None}


def load_assist09_s(args):
    dataset_path = 'Dataset/assist2009/processed_data.csv'
    ratio = [0.8, 0.2]
    args.s_num = 117
    args.q_num = 117
    data_path = dataset_path
    question_num = args.q_num
    max_seq_len = args.max_seq_len
    ratio = ratio
    data_shuffle = False
    seq_len_list = []
    question_list = []
    answer_list = []
    feature_list = []
    dataset_dir = data_path.split('/')
    dataset_dir = '/'.join(dataset_dir[:-1])
    print(dataset_dir)
    problem_set = set()
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            idx = i
            line = lines[i]
            if idx % 3 == 0:
                seq_len = int(line)
                # if seq_len < 20:
                #     i += 2
                #     continue
                seq_len_list.append(min(seq_len, max_seq_len))


            elif idx % 3 == 1:
                question_seq = line.split(',')
                question_seq = [int(s) for s in question_seq][:max_seq_len]
                for q in question_seq:
                    problem_set.add(q)

                question_list.append(question_seq)
            else:
                answer_seq = line.split(',')
                answer_seq = [int(s) for s in answer_seq][:max_seq_len]
                answer_list.append(answer_seq)
                # feature_list.append(
                #     [2 * (q-1) + answer_seq[idx]  for idx, q in enumerate(skill_seq)]
                # )
                # feature_list.append([answer_seq[idx] * question_num + q for idx, q in enumerate(question_seq)])

    f.close()

    print(f'Dataset Question Num: {problem_set.__len__()}')
    set_values = [i for i in range(1, len(problem_set) + 1)]
    problem_dict = dict(zip(problem_set, set_values))
    print(problem_dict)

    for question_seq in question_list:
        for idx, q in enumerate(question_seq):
            question_seq[idx] = problem_dict[q]
    for question_seq in question_list:
        feature_list.append([answer_seq[idx] * question_num + q for idx, q in enumerate(question_seq)])
        for idx, q in enumerate(question_seq):
            assert q <= question_num
            assert q > 0
    for i in range(len(feature_list)):
        f_seq = feature_list[i]
        q_seq = question_list[i]
        a_seq = answer_list[i]
        for j in range(len(f_seq)):
            assert f_seq[j] == a_seq[j] * question_num + q_seq[j]
            f_seq[j] = a_seq[j] * question_num + q_seq[j]
    '''
        Do some check here
        You just never can be too cautious
    '''

    def check():
        for seq in question_list:
            assert max(seq) <= args.q_num
            assert min(seq) > 0

        for seq in answer_list:
            assert max(seq) <= 1
            assert min(seq) >= 0

    check()

    def pad(target, value, max_len):
        for idx, pad_seq in enumerate(target):
            pad_len = max_len - len(pad_seq)
            target[idx] = pad_seq + [value for i in range(pad_len)]

    pad(question_list, 0, args.max_seq_len)

    pad(answer_list, -1, args.max_seq_len)

    questions = np.array(question_list)
    seq_len = np.array(seq_len_list)
    skills = np.array(question_list)
    answers = np.array(answer_list)

    kt_dataset = KTDataset(q_num=args.q_num, s_num=args.s_num, questions=questions,
                           skills=skills, answers=answers, seq_len=seq_len, max_seq_len=args.max_seq_len)

    dataset_size = len(kt_dataset)
    print(f'Dataset Size: {dataset_size}')

    train_size = int(ratio[0] * dataset_size)
    test_size = dataset_size - train_size

    train_set, test_set = random_split(kt_dataset, [train_size, test_size])
    if args.data_augment:
        print('We are augmenting the dataset')
        print('Before augmentation the size of trainset is ', len(train_set))
        train_set = train_set.dataset
        train_set.print()
        train_set.augment()
        print('After Augmentation: -----------------------')
        train_set.print()
        print(type(train_set))
        train_set, _ = random_split(train_set, [len(train_set), 0])
        print('After the augmentation the size of trainset is ', len(train_set))
    if False:
        test_set.purify()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=data_shuffle)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=data_shuffle)

    print(train_loader)
    print(test_loader)

    return {'train_loader': train_loader, 'test_loader': test_loader, 'qs_matrix': None}


def load_assist17_s(args):
    pass


def load_ednet(args):
    data_dir = 'Dataset\\' + args.dataset
    data = np.load(os.path.join(data_dir, args.dataset + '.npz'))
    y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']

    args.s_num, args.q_num = data['skill_num'], data['problem_num']

    train_data, test_data = train_test_split([y, skill, problem, real_len])  # [y, skill, pro, real_len]
    train_y, train_skill, train_problem, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
    test_y, test_skill, test_problem, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
    args.q_num = max(np.max(train_problem), np.max(test_problem))
    train_set = KTDataset(args.q_num, args.s_num, train_problem, train_skill, train_y, train_real_len,
                          max_seq_len=args.max_seq_len)
    test_set = KTDataset(args.q_num, args.s_num, test_problem, test_skill, test_y, test_real_len,
                         max_seq_len=args.max_seq_len)
    if args.data_augment:
        print('Use Data Augmentation')
        train_set.augment()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=args.shuffle)
    '''
        Modification Required
        need a better way to load qs_matrix cuz the original file is like a crap of shit 
    '''
    return {'train_loader': train_loader, 'test_loader': test_loader, 'qs_matrix': None}


def load_buaa(args):
    print("Loading Beihang1819 Dataset")
    print(args)
    from Dataset.beihang.analyse import reload_all, load, load_qmatrix
    q_matrix = load_qmatrix()
    q_matrix = torch.IntTensor(q_matrix)
    q_matrix = torch.cat([torch.zeros([1, q_matrix.shape[1]]), q_matrix])

    Q_buaa, A_buaa, L_buaa = load()

    y = A_buaa
    skill = Q_buaa
    problem = Q_buaa
    real_len = L_buaa
    q_num = np.max(problem)

    args.q_num = q_num
    args.s_num = 15
    print(args.q_num)
    train_data, test_data = train_test_split([y, skill, problem, real_len])  # [y, skill, pro, real_len]
    train_y, train_skill, train_problem, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
    test_y, test_skill, test_problem, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
    train_set = KTDataset(q_num=args.q_num,
                          s_num=args.s_num,
                          questions=train_problem,
                          skills=None,
                          answers=train_y,
                          seq_len=train_real_len,
                          max_seq_len=args.max_seq_len)

    test_set = KTDataset(q_num=args.q_num,
                         s_num=args.s_num,
                         questions=test_problem,
                         skills=None,
                         answers=test_y,
                         seq_len=test_real_len,
                         max_seq_len=args.max_seq_len)
    if args.data_augment:
        print('Use Data Augmentation')
        train_set.augment()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=args.shuffle)
    '''
        Modification Required
        need a better way to load qs_matrix cuz the original file is like a crap of shit 
    '''
    return {'train_loader': train_loader, 'test_loader': test_loader, 'qs_matrix': q_matrix}


def load_junyi(args):
    ratio = [0.8, 0.2]

    max_seq_len = args.max_seq_len
    ratio = ratio
    data_shuffle = False
    seq_len_list = []
    question_list = []
    answer_list = []
    feature_list = []
    problem_set = set()
    with open('Dataset/junyi/student_log_kt_1000_tl.txt', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            idx = i
            line = lines[i]
            if idx % 3 == 0:
                seq_len = int(line)
                # if seq_len < 20:
                #     i += 2
                #     continue
                seq_len_list.append(min(seq_len, max_seq_len))


            elif idx % 3 == 1:
                question_seq = line.split(',')
                question_seq = [int(s) for s in question_seq][:max_seq_len]
                for q in question_seq:
                    problem_set.add(q)
                    assert type(q) is not str

                question_list.append(question_seq)
            else:
                answer_seq = line.split(',')
                answer_seq = [int(s) for s in answer_seq][:max_seq_len]
                answer_list.append(answer_seq)
                # feature_list.append(
                #     [2 * (q-1) + answer_seq[idx]  for idx, q in enumerate(skill_seq)]
                # )
                # feature_list.append([answer_seq[idx] * question_num + q for idx, q in enumerate(question_seq)])

    f.close()

    print(f'Junyi Dataset Question Num: {problem_set.__len__()}')
    print(f'Min Question ID {min(problem_set)} , Max Question ID {max(problem_set)}')
    set_values = [i for i in range(1, len(problem_set) + 1)]

    problem_dict = dict(zip(problem_set, set_values))
    print(problem_dict)
    with open('Dataset/junyi/pid_map', 'w') as json_out:
        json.dump(problem_dict, json_out)
    question_num = len(problem_set)
    args.q_num = question_num
    args.s_num = question_num
    for question_seq in question_list:
        for idx, q in enumerate(question_seq):
            question_seq[idx] = problem_dict[q]

    for i in range(len(question_list)):

        q_seq = question_list[i]
        a_seq = answer_list[i]
        f_seq = [a_seq[idx] * question_num + q for idx, q in enumerate(q_seq)]
        feature_list.append(f_seq)
        for j in range(len(f_seq)):
            assert f_seq[j] == a_seq[j] * question_num + q_seq[j]
    '''
        Do some check here
        You just never can be too cautious
    '''

    def check():
        for seq in question_list:
            assert max(seq) <= args.q_num
            assert min(seq) > 0

        for seq in answer_list:
            assert max(seq) <= 1
            assert min(seq) >= 0

    check()

    def pad(target, value, max_len):
        for idx, pad_seq in enumerate(target):
            pad_len = max_len - len(pad_seq)
            target[idx] = pad_seq + [value for i in range(pad_len)]

    pad(question_list, 0, args.max_seq_len)

    pad(answer_list, -1, args.max_seq_len)

    questions = np.array(question_list)
    seq_len = np.array(seq_len_list)
    skills = np.array(question_list)
    answers = np.array(answer_list)

    kt_dataset = KTDataset(q_num=args.q_num, s_num=args.s_num, questions=questions,
                           skills=skills, answers=answers, seq_len=seq_len, max_seq_len=args.max_seq_len)

    dataset_size = len(kt_dataset)
    print(f'Dataset Size: {dataset_size}')

    train_size = int(ratio[0] * dataset_size)
    test_size = dataset_size - train_size

    train_set, test_set = random_split(kt_dataset, [train_size, test_size])
    if args.data_augment:
        print('We are augmenting the dataset')
        print('Before augmentation the size of trainset is ', len(train_set))
        train_set = train_set.dataset
        train_set.print()
        train_set.augment()
        print('After Augmentation: -----------------------')
        train_set.print()
        print(type(train_set))
        train_set, _ = random_split(train_set, [len(train_set), 0])
        print('After the augmentation the size of trainset is ', len(train_set))
    if False:
        test_set.purify()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=data_shuffle)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=data_shuffle)

    print(train_loader)
    print(test_loader)

    return {'train_loader': train_loader, 'test_loader': test_loader, 'qs_matrix': None}


def load_junyi2(args):
    ratio = [0.8, 0.2]

    max_seq_len = args.max_seq_len
    ratio = ratio
    data_shuffle = False

    import pickle
    with open('Dataset/junyi/ex2exid.pkl', 'rb') as f:
        ex2exid = pickle.load(f)

    with open('Dataset/junyi/exid2ex.pkl', 'rb') as f:
        exid2ex = pickle.load(f)

    args.q_num = len(exid2ex)
    question_num = args.q_num

    with open('Dataset/junyi/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    question_list = data['q_seq']
    answer_list = data['a_seq']
    feature_list = []
    seq_len_list = []
    for i in range(len(question_list)):

        q_seq = question_list[i]
        a_seq = answer_list[i]
        f_seq = [a_seq[idx] * question_num + q for idx, q in enumerate(q_seq)]
        feature_list.append(f_seq)
        seq_len_list.append(len(f_seq))
        for j in range(len(f_seq)):
            assert f_seq[j] == a_seq[j] * question_num + q_seq[j]

    def pad(target, value, max_len):
        for idx, pad_seq in enumerate(target):
            if len(target[idx]) > args.max_seq_len:
                print(len(target[idx]))
                target[idx] = target[idx][:args.max_seq_len]
            else:
                pad_len = max_len - len(pad_seq)
                target[idx] = pad_seq + [value for i in range(pad_len)]

    pad(question_list, 0, args.max_seq_len)

    pad(answer_list, -1, args.max_seq_len)

    questions = np.array(question_list)
    seq_len = np.array(seq_len_list)
    skills = np.array(question_list)
    answers = np.array(answer_list)

    kt_dataset = KTDataset(q_num=args.q_num, s_num=args.s_num, questions=questions,
                           skills=skills, answers=answers, seq_len=seq_len, max_seq_len=args.max_seq_len)

    dataset_size = len(kt_dataset)
    print(f'Dataset Size: {dataset_size}')

    train_size = int(ratio[0] * dataset_size)
    test_size = dataset_size - train_size

    train_set, test_set = random_split(kt_dataset, [train_size, test_size])
    if args.data_augment:
        print('We are augmenting the dataset')
        print('Before augmentation the size of trainset is ', len(train_set))
        train_set = train_set.dataset
        train_set.print()
        train_set.augment()
        print('After Augmentation: -----------------------')
        train_set.print()
        print(type(train_set))
        train_set, _ = random_split(train_set, [len(train_set), 0])
        print('After the augmentation the size of trainset is ', len(train_set))
    if False:
        test_set.purify()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=data_shuffle)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=data_shuffle)

    return {'train_loader': train_loader, 'test_loader': test_loader, 'qs_matrix': None}


def load_ednet_qs(args):
    ratio = [0.8, 0.2]
    data_shuffle = False
    with open('Dataset/ednet/pro_id_dict.txt', 'r') as f:
        pro_id_dict = eval(f.read())
        print(pro_id_dict)
        '''
        人为规定问题编号始终从1开始
        '''
        for k, v in pro_id_dict.items():
            pro_id_dict[k] = v + 1
        args.q_num = len(pro_id_dict)
        q_id_list = pro_id_dict.values()

        args.q_num = max(q_id_list)
        assert min(q_id_list) > 0
    with open('Dataset/ednet/pro_skill_dict.txt', 'r') as f:
        pro_skill_dict = eval(f.read())
        print(pro_skill_dict)
        args.s_num = -1
        for k, v in pro_skill_dict.items():
            '''
            skills are in the format of 'a,b,c'
            '''
            skills = v.split(';')
            skills = [int(s) for s in skills]
            for s in skills:
                assert s > 0
            args.s_num = max(args.s_num, max(skills))
    print(args.s_num)
    '''
    Calculate qs_matrix
    '''
    qs_matrix = np.zeros((args.q_num + 1, args.s_num + 1))
    print(qs_matrix.shape)
    for k, v in pro_skill_dict.items():
        '''
        skills are in the format of 'a,b,c'
        '''
        skills = v.split(';')
        skills = [int(s) for s in skills]
        q_id = pro_id_dict[k]
        for s_id in skills:
            qs_matrix[q_id, s_id] = 1
    print(qs_matrix)
    question_list = []
    answer_list = []
    seq_len_list = []
    with open('Dataset/ednet/data.txt', 'r') as f:
        lines = f.readlines()
        '''
        Sample Data Format
        [10]
        [101, 77, 86, 113, 99, 77, 123, 134, 106, 119]
        [4887, 8274, 9335, 6163, 5504, 6424, 4431, 5087, 4644, 4756]
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        
        
        '''
        for idx, line in enumerate(lines):
            line = line[1:-2]
            if idx % 4 == 0:
                seq_len = int(line)
                seq_len_list.append(min(seq_len, args.max_seq_len))
            elif idx % 4 == 1:
                pass
            elif idx % 4 == 2:
                q_seq = line.split(',')
                q_seq = [int(q) for q in q_seq]
                question_list.append(q_seq)
            elif idx % 4 == 3:
                a_seq = line.split(',')
                a_seq = [int(a) for a in a_seq]
                answer_list.append(a_seq)

    def pad(target, value, max_len):
        for idx, pad_seq in enumerate(target):
            if len(target[idx]) > args.max_seq_len:
                print(len(target[idx]))
                target[idx] = target[idx][:args.max_seq_len]
            else:
                pad_len = max_len - len(pad_seq)
                target[idx] = pad_seq + [value for i in range(pad_len)]

    pad(question_list, 0, args.max_seq_len)

    pad(answer_list, -1, args.max_seq_len)

    questions = np.array(question_list)
    seq_len = np.array(seq_len_list)
    skills = np.array(question_list)
    answers = np.array(answer_list)

    kt_dataset = KTDataset(q_num=args.q_num, s_num=args.s_num, questions=questions,
                           skills=skills, answers=answers, seq_len=seq_len, max_seq_len=args.max_seq_len)

    dataset_size = len(kt_dataset)
    print(f'Dataset Size: {dataset_size}')

    train_size = int(ratio[0] * dataset_size)
    test_size = dataset_size - train_size

    train_set, test_set = random_split(kt_dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=data_shuffle)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=data_shuffle)

    return {'train_loader': train_loader, 'test_loader': test_loader, 'qs_matrix': qs_matrix}


def load_dataset(args):
    if args.dataset == 'assist09-q':
        return load_assist09_q(args)
    elif args.dataset == 'assist09-s':
        return load_assist09_s(args)
    elif args.dataset == 'assist17-s':
        return load_assist17_s(args)
    elif args.dataset == 'beihang':
        return load_buaa(args)
    elif args.dataset == 'buaa18a':
        pass
    elif args.dataset == 'buaa18s':
        pass
    elif args.dataset == 'ednet_qs':
        return load_ednet_qs(args)
    elif args.dataset == 'ednet':
        return load_ednet(args)
    elif args.dataset == 'junyi':
        return load_junyi(args)
    elif args.dataset == 'junyi2':
        return load_junyi2(args)
    else:
        pass

    ratio = [0.7, 0.25, 0.05]
    # if dataset_info['s_graph'] != 'False':
    #     s_graph = np.load(dataset_info['s_graph'])
    pass


def load_model(args):
    def load_checkpoint(model, optimizer, checkpoint_PATH):
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('Loading Checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
        return model, optimizer

    if args.model == 'DKT':
        from KT.models.DKT import DKT
        model = DKT(feature_dim=2 * args.q_num + 1,
                    embed_dim=args.embed_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.q_num)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.checkpoint_dir is not None:
            model, optimizer = load_checkpoint(model, optimizer, args.checkpoint_dir)
        return model, optimizer
    if args.model == 'DKT_AUG':
        from KT.models.DKT import DKT_AUG
        model = DKT_AUG(
            feature_dim=2 * args.q_num + 1,
            skill_num=args.s_num,
            question_num=args.q_num,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.q_num,
            qs_matrix=args.qs_matrix,
            dropout=0.2,
            bias=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.checkpoint_dir is not None:
            model, optimizer = load_checkpoint(model, optimizer, args.checkpoint_dir)
        return model, optimizer
    if args.model == 'DKT_PEBG':
        from KT.models.DKT import DKT_PEBG
        model = DKT_PEBG(pro_num=args.q_num + 1,
                         skill_num=args.s_num,
                         hidden_dim=args.hidden_dim,
                         embed_dim=args.embed_dim,
                         output_dim=args.q_num)
        if args.pretrain in ['load', 'scratch']:
            model.load_pretrain_embedding(args.pretrain_embed_file)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.checkpoint_dir is not None:
            model, optimizer = load_checkpoint(model, optimizer, args.checkpoint_dir)
        return model, optimizer
    elif args.model == 'SAKT':
        from models.SAKT import SAKT
        model = SAKT(q_num=args.q_num, seq_len=args.max_seq_len, embed_dim=args.embed_dim, heads=5,
                     dropout=args.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return model, optimizer
    elif args.model == 'DKVMN':
        from KT.models.DKVMN import DKVMN
        q_embed_dim = args.embed_dim
        qa_embed_dim = args.embed_dim
        final_fc_dim = 50
        model = DKVMN(n_question=args.q_num,
                      batch_size=args.batch_size,
                      q_embed_dim=50,
                      qa_embed_dim=100,
                      memory_size=20,
                      memory_key_state_dim=50,
                      memory_value_state_dim=100,
                      final_fc_dim=final_fc_dim)
        model.init_embeddings()
        model.init_params()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return model, optimizer
    elif args.model == 'DKVMN_RE':
        from KT.models.DKVMN_RE import DKVMN_RE
        q_embed_dim = args.embed_dim
        qa_embed_dim = args.embed_dim
        final_fc_dim = 50
        model = DKVMN_RE(n_question=args.q_num,
                         batch_size=args.batch_size,
                         q_embed_dim=q_embed_dim,
                         qa_embed_dim=qa_embed_dim,
                         memory_size=args.s_num,
                         memory_key_state_dim=q_embed_dim,
                         memory_value_state_dim=qa_embed_dim,
                         final_fc_dim=final_fc_dim)
        model.set_qs_matrix(args.qs_matrix)
        model.init_embeddings()
        model.init_params()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return model, optimizer
    elif args.model == 'AKT':
        from KT.models.AKT import AKT
        print(args.q_num)
        model = AKT(n_question=args.q_num, n_pid=args.q_num, n_blocks=1, d_model=256,
                    dropout=0.05, kq_same=1, model_type='akt', l2=1e-5)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return model, optimizer
    elif args.model == 'SAINT':
        from KT.models.SAINT import saint
        model = saint(dim_model=128,
                      num_en=6,
                      num_de=6,
                      heads_en=8,
                      heads_de=8,
                      total_ex=total_ex,
                      total_cat=total_cat,
                      total_in=total_in,
                      seq_len=args.seq_len
                      )
    elif args.model == 'GKT':
        # if args.graph_type == 'MHA':
        #     graph_model = MultiHeadAttention(args.edge_types, concept_num, args.emb_dim, args.attn_dim, dropout=args.dropout)
        # elif args.graph_type == 'VAE':
        #     graph_model = VAE(args.emb_dim, args.vae_encoder_dim, args.edge_types, args.vae_decoder_dim, args.vae_decoder_dim, concept_num,
        #                       edge_type_num=args.edge_types, tau=args.temp, factor=args.factor, dropout=args.dropout, bias=args.bias)
        #     vae_loss = VAELoss(concept_num, edge_type_num=args.edge_types, prior=args.prior, var=args.var)
        #     if args.cuda:
        #         vae_loss = vae_loss.cuda()
        # if args.cuda and args.graph_type in ['MHA', 'VAE']:
        #     graph_model = graph_model.cuda()
        from models.GKT import GKT
        graph_model = None
        # graph = build_dense_graph(question_num)
        graph = None
        model = GKT(args.s_num, args.hidden_dim, args.embedding_dim, args.edge_types, args.graph_type, graph=graph,
                    graph_model=graph_model,
                    dropout=args.dropout, has_cuda=torch.cuda.is_available())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return model, optimizer
    elif args.model == 'QGKT':
        from models.QGKT import QGKT

        def s_graph_gen(s_n):
            s_graph = np.ones((s_n, s_n))
            s_graph = s_graph / (s_n - 1)
            np.fill_diagonal(s_graph, 0)

            return s_graph

        model = QGKT(question_num=args.q_num, skill_num=args.s_num, hidden_dim=args.hidden_dim,
                     embedding_dim=args.embed_dim,
                     qs_matrix=args.qs_matrix, s_graph=s_graph_gen(args.s_num))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return model, optimizer
    else:
        model = None
        optimizer = None
        return model, optimizer


if __name__ == '__main__':
    load_buaa()
