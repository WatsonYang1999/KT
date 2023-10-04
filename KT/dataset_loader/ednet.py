import numpy as np
import os
from KT.KTDataloader import KTDataset
from KT.utils import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split, Subset


def load_ednet(args):
    data_dir = os.path.join('Dataset',args.dataset)
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

def load_ednet(args):
    data_dir = os.path.join('Dataset',args.dataset)
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

def load_ednet_qs(args):
    ratio = [0.8, 0.2]
    data_shuffle = False
    with open('Dataset/ednet/pro_id_dict.txt', 'r') as f:
        pro_id_dict = eval(f.read())
        # print(pro_id_dict)
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
        # print(pro_skill_dict)
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
    # print(qs_matrix.shape)
    for k, v in pro_skill_dict.items():
        '''
        skills are in the format of 'a,b,c'
        '''
        skills = v.split(';')
        skills = [int(s) for s in skills]
        q_id = pro_id_dict[k]
        for s_id in skills:
            qs_matrix[q_id, s_id] = 1
    # print(qs_matrix)
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
    # print(f'Dataset Size: {dataset_size}')

    train_size = int(ratio[0] * dataset_size)
    test_size = dataset_size - train_size

    train_set, test_set = random_split(kt_dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=data_shuffle)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=data_shuffle)

    return {'train_loader': train_loader, 'test_loader': test_loader, 'qs_matrix': qs_matrix}
