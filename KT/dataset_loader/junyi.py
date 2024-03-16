import json

import numpy as np
from torch.utils.data import random_split

from kt_dataset import KTDataset
from util.kt_util import pad


def load_junyi(args):
    print('loading junyi dataset')
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

    qs_matrix = None
    return train_set, test_set, qs_matrix


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

    # todo : load qs_matrix
    qs_matrix = None
    return train_set, test_set, qs_matrix
