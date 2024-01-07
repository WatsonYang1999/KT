import json
import os
import random
from datetime import datetime

import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from KT.KTDataloader import KTDataset, KTDataset_SA
from KT.util.decorators import timing_decorator, suit_directory_decorator

import os


def pad(target, value, max_len):
    for idx, pad_seq in enumerate(target):
        if len(target[idx]) > max_len:
            target[idx] = target[idx][:max_len]
        else:
            pad_len = max_len - len(pad_seq)
            target[idx] = pad_seq + [value for i in range(pad_len)]


def cut_seq(seqs, max_len, qs_mapping, sq_mapping, min_seq_len=5, skill_select_strategy='first_select',
            dataset_scale='full'):
    '''
    :param seqs:seqs[uid] = {"question": q_seq, "result": r_seq, "ca": ca_seq, "sa": sa_seq}
    '''
    q_seq_list = []
    y_seq_list = []
    s_seq_list = []
    seq_len_list = []
    q_padding = -1
    y_padding = -1
    s_padding = -1

    def pad(seq, max_len, pad_value):
        if len(seq) < max_len:
            return seq + [pad_value for _ in range(0, max_len - len(seq))]
        return seq[:max_len]

    from tqdm import tqdm
    i = 0
    mini_size = 3000

    break_into_multi_seqs = False

    for uid, seqs in tqdm(seqs.items(), desc="Processing", unit="iteration", ncols=80):

        q_seq = seqs['question']
        a_seq = seqs['result']
        seq_len = len(q_seq)

        if seq_len < min_seq_len:
            continue

        def select_skill(question):
            if skill_select_strategy == 'first_select':
                return list(qs_mapping[question])[0]

        if skill_select_strategy == 'total_random':

            random_skill_list = random.choices(list(sq_mapping.keys()), k=len(q_seq))
            s_seq = random_skill_list

        else:
            s_seq = [select_skill(q) for q in q_seq]

        if break_into_multi_seqs:
            while seq_len > max_len:
                q_seq_list.append(q_seq[:max_len])
                q_seq = q_seq[max_len:]
                y_seq_list.append(a_seq[:max_len])
                a_seq = a_seq[max_len:]
                s_seq_list.append(s_seq[:max_len])
                s_seq = s_seq[max_len:]
                seq_len_list.append(max_len)
                seq_len -= max_len

            q_seq_list.append(pad(q_seq, max_len, q_padding))
            y_seq_list.append(pad(a_seq, max_len, y_padding))
            s_seq_list.append(pad(s_seq, max_len, s_padding))
            seq_len_list.append(min(seq_len, max_len))
            i += 1
        else:
            q_seq_list.append(pad(q_seq, max_len, q_padding))
            y_seq_list.append(pad(a_seq, max_len, y_padding))
            s_seq_list.append(pad(s_seq, max_len, s_padding))
            seq_len_list.append(min(seq_len, max_len))
            i += 1
        if dataset_scale == 'mini':
            if i > mini_size:
                break
    return np.array(q_seq_list, dtype=int), np.array(s_seq_list, dtype=int), np.array(y_seq_list,
                                                                                      dtype=float), np.array(
        seq_len_list, dtype=int)


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


def reformat_datatime(dt: datetime):
    formatted_time = dt.strftime("%y-%m-%d_%H-%M-%S")
    return formatted_time


def check_gpu_memory_allocated():
    device = 'cuda'
    allocated_memory = torch.cuda.memory_allocated(device)  # Convert bytes to gigabytes
    peak_allocated_memory = torch.cuda.max_memory_allocated(device)

    print(f"Initial GPU Memory Allocated: {allocated_memory / (1024 ** 2):.2f} MB")
    print(f"GPU Peak Memory Allocated: {peak_allocated_memory / (1024 ** 2):.2f} MB")


# divide train test set
def train_test_split(data, split=0.8, random_split=True):
    n_samples = data[0].shape[0]
    if random_split:
        indices = np.random.permutation(n_samples)
    else:
        indices = [i for i in range(0, n_samples)]
    split_point = int(n_samples * split)
    train_data, test_data = [], []
    for d in data:
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]
        train_data.append(d[train_idx])
        test_data.append(d[test_idx])
    return train_data, test_data


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
    qs_matrix = None
    return train_set, test_set, qs_matrix


@suit_directory_decorator()
@timing_decorator
def load_ednet_re(args):
    data_path = 'Dataset/ednet-re/data'

    skill_select_strategy = 'first_select'
    # skill_select_strategy = 'total_random'
    # skill_select_strategy = 'related_random'
    # skill_select_strategy = 'most_frequent'
    # dataset_scale = 'mini'
    dataset_scale = 'full'

    load_custom = args.custom_data
    custom_dataset = os.path.join(data_path, 'custom_data', 'custom_data_1.npz')
    preprocessed_dataset = 'ednet-re-' + dataset_scale + '_' + skill_select_strategy + '.npz'

    from KT.dataset_loader.ednet_re import build_user_sequences, load_qs_relations
    qs_mapping, sq_mapping = load_qs_relations()
    custom_task1_path = 'Dataset/ednet-re/data/custom_data/custom_task_1.csv'
    if load_custom:
        custom_seqs = build_user_sequences(custom_task1_path)

        custom_q, custom_s, custom_y, custom_real_len = cut_seq(custom_seqs, args.max_seq_len, min_seq_len=0,
                                                                skill_select_strategy=skill_select_strategy,
                                                                dataset_scale=dataset_scale)

        np.savez(
            custom_dataset,
            q_num=len(qs_mapping),
            s_num=len(sq_mapping),
            q=custom_q,
            s=custom_s,
            y=custom_y,
            seq_len=custom_real_len,
            qs_mapping=qs_mapping,
            sq_mapping=sq_mapping
        )
    always_load_from_preprocess = True
    if not os.path.exists(os.path.join(data_path, preprocessed_dataset)) or always_load_from_preprocess:

        train_task1_path = 'Dataset/ednet-re/data/train_data/train_task_1_2.csv'
        test_task1_public_path = 'Dataset/ednet-re/data/test_data/test_public_answers_task_1_2.csv'
        skill_metadata_path = 'Dataset/ednet-re/data/metadata/subject_metadata.csv'

        total_data_path = 'Dataset/ednet-re/data/total_time_ordered.csv'

        # train_seqs = build_user_sequences(train_task1_path)
        # test_public_seqs = build_user_sequences(test_task1_public_path)
        total_seqs = build_user_sequences(total_data_path)
        args.q_num = len(qs_mapping)
        args.s_num = len(sq_mapping)
        # train_q, train_s, train_y, train_real_len = cut_seq(train_seqs, args.max_seq_len)
        # assert train_q.shape == train_y.shape
        #
        # test_q, test_s, test_y, test_real_len = cut_seq(test_public_seqs, args.max_seq_len)
        #
        # assert test_q.shape == test_y.shape
        # q_merged = np.vstack((train_q, test_q))
        # s_merged = np.vstack((train_s, test_s))
        # y_merged = np.vstack((train_y, test_y))
        # real_len_merged = np.hstack((train_real_len, test_real_len))
        q_merged, s_merged, y_merged, real_len_merged = cut_seq(total_seqs, args.max_seq_len, qs_mapping, sq_mapping)
        train_data, test_data = train_test_split([y_merged, s_merged, q_merged, real_len_merged], random_split=False)
        train_y, train_s, train_q, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
        test_y, test_s, test_q, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
        if not always_load_from_preprocess:
            np.savez(
                os.path.join(data_path, preprocessed_dataset),
                q_num=len(qs_mapping),
                s_num=len(sq_mapping),
                q=q_merged,
                s=s_merged,
                y=y_merged,
                seq_len=real_len_merged,
                qs_mapping=qs_mapping,
                sq_mapping=sq_mapping
            )

    else:
        data = np.load(os.path.join(data_path, preprocessed_dataset), allow_pickle=True)
        y, skill, problem, real_len = data['y'], data['s'], data['q'], data['seq_len']

        args.s_num, args.q_num = int(data['s_num']), int(data['q_num'])

        train_data, test_data = train_test_split([y, skill, problem, real_len], random_split=False)
        train_y, train_s, train_q, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
        test_y, test_s, test_q, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
        # very ugly idea
        qs_mapping = eval(data['qs_mapping'].__str__())
        sq_mapping = eval(data['sq_mapping'].__str__())
        if load_custom:
            custom_data = np.load(os.path.join(data_path, custom_dataset), allow_pickle=True)
            test_y, test_s, test_q, test_real_len = custom_data['y'], custom_data['s'], custom_data['q'], custom_data[
                'seq_len']

    train_set = KTDataset(args.q_num, args.s_num, train_q, train_s, train_y, train_real_len, args.max_seq_len,
                          remapping=True,
                          qs_mapping=qs_mapping,
                          sq_mapping=sq_mapping,
                          )

    test_set = KTDataset(args.q_num, args.s_num, test_q, test_s, test_y, test_real_len, args.max_seq_len,
                         remapping=True,
                         qs_mapping=qs_mapping,
                         sq_mapping=sq_mapping,
                         )

    print("Done Loading Datasets")
    qs_matrix = train_set.get_qs_matrix()
    return train_set, test_set, qs_matrix


@suit_directory_decorator()
@timing_decorator
def load_assist09_q(args):
    data_path = 'Dataset/assist2009'

    skill_select_strategy = 'first_select'
    # skill_select_strategy = 'total_random'
    # skill_select_strategy = 'related_random'
    # skill_select_strategy = 'most_frequent'
    # dataset_scale = 'mini'
    dataset_scale = 'full'

    load_custom = args.custom_data
    custom_dataset = os.path.join(data_path, 'custom_data', 'custom_data_1.npz')
    preprocessed_dataset = 'assist09-re-' + dataset_scale + '_' + skill_select_strategy + '.npz'

    from Dataset.assist2009.preprocess import build_user_sequences, load_qs_relation
    qs_mapping, sq_mapping = load_qs_relation()
    if load_custom:
        custom_seqs = build_user_sequences()

        custom_q, custom_s, custom_y, custom_real_len = cut_seq(custom_seqs, args.max_seq_len, qs_mapping, sq_mapping,
                                                                min_seq_len=0,
                                                                skill_select_strategy=skill_select_strategy,
                                                                dataset_scale=dataset_scale)

        np.savez(
            custom_dataset,
            q_num=len(qs_mapping),
            s_num=len(sq_mapping),
            q=custom_q,
            s=custom_s,
            y=custom_y,
            seq_len=custom_real_len,
            qs_mapping=qs_mapping,
            sq_mapping=sq_mapping
        )
    always_load_from_preprocess = True
    if not os.path.exists(os.path.join(data_path, preprocessed_dataset)) or always_load_from_preprocess:

        total_seqs = build_user_sequences()

        args.q_num = len(qs_mapping)
        args.s_num = len(sq_mapping)
        # train_q, train_s, train_y, train_real_len = cut_seq(train_seqs, args.max_seq_len)
        # assert train_q.shape == train_y.shape
        #
        # test_q, test_s, test_y, test_real_len = cut_seq(test_public_seqs, args.max_seq_len)
        #
        # assert test_q.shape == test_y.shape
        # q_merged = np.vstack((train_q, test_q))
        # s_merged = np.vstack((train_s, test_s))
        # y_merged = np.vstack((train_y, test_y))
        # real_len_merged = np.hstack((train_real_len, test_real_len))

        q_merged, s_merged, y_merged, real_len_merged = cut_seq(total_seqs, args.max_seq_len, qs_mapping, sq_mapping,
                                                                min_seq_len=0,
                                                                skill_select_strategy=skill_select_strategy,
                                                                dataset_scale=dataset_scale)

        train_data, test_data = train_test_split([y_merged, s_merged, q_merged, real_len_merged], random_split=False)
        train_y, train_s, train_q, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
        test_y, test_s, test_q, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
        if not always_load_from_preprocess:
            np.savez(
                os.path.join(data_path, preprocessed_dataset),
                q_num=len(qs_mapping),
                s_num=len(sq_mapping),
                q=q_merged,
                s=s_merged,
                y=y_merged,
                seq_len=real_len_merged,
                qs_mapping=qs_mapping,
                sq_mapping=sq_mapping
            )

    else:
        data = np.load(os.path.join(data_path, preprocessed_dataset), allow_pickle=True)
        y, skill, problem, real_len = data['y'], data['s'], data['q'], data['seq_len']

        args.s_num, args.q_num = int(data['s_num']), int(data['q_num'])

        train_data, test_data = train_test_split([y, skill, problem, real_len], random_split=False)
        train_y, train_s, train_q, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
        test_y, test_s, test_q, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
        # very ugly idea
        qs_mapping = eval(data['qs_mapping'].__str__())
        sq_mapping = eval(data['sq_mapping'].__str__())
        if load_custom:
            custom_data = np.load(os.path.join(data_path, custom_dataset), allow_pickle=True)
            test_y, test_s, test_q, test_real_len = custom_data['y'], custom_data['s'], custom_data['q'], custom_data[
                'seq_len']

    train_set = KTDataset(args.q_num, args.s_num, train_q, train_s, train_y, train_real_len, args.max_seq_len,
                          remapping=True,
                          qs_mapping=qs_mapping,
                          sq_mapping=sq_mapping,
                          )

    test_set = KTDataset(args.q_num, args.s_num, test_q, test_s, test_y, test_real_len, args.max_seq_len,
                         remapping=True,
                         qs_mapping=qs_mapping,
                         sq_mapping=sq_mapping,
                         )
    print(f'Done Loading Assist09-Question-Level , train_size: {len(train_set)} test_size: {len(test_set)}')
    qs_matrix = train_set.get_qs_matrix()

    return train_set, test_set, qs_matrix


def load_dummy_dataset(args):
    q_num = 4
    s_num = 3
    max_seq_len = 10
    q_list = [21, 22, 23, 24]
    s_list = [5, 6, 7]

    qs_mapping = {
        21: {5},
        22: {6},
        23: {5, 6},
        24: {6, 7},
    }

    sq_mapping = {
        5: {21, 23},
        6: {22, 23},
        7: {24}
    }

    train_q = torch.IntTensor(
        [[21, 22, 23, 24, -1, -1, -1, -1, -1, -1]]
    )
    test_q = torch.IntTensor(
        [[24,23, 22,21, -1, -1, -1, -1, -1, -1]]
    )
    train_s = torch.IntTensor([[5 , 6, 7, 5, -1, -1, -1, -1, -1, -1]])
    test_s = torch.IntTensor([[5, 6, 5, 5, -1, -1, -1, -1, -1, -1]])
    train_y = torch.IntTensor([[1, 0, 1, 0,  -1, -1,  -1, -1,  -1, -1]])
    test_y = torch.IntTensor([[0, 1, 0, 1, -1, -1,  -1, -1,  -1, -1]])
    train_real_len = torch.IntTensor([4])
    test_real_len = torch.IntTensor([4])
    train_set = KTDataset(q_num, s_num, train_q, train_s, train_y, train_real_len, max_seq_len,
                          remapping=True,
                          qs_mapping=qs_mapping,
                          sq_mapping=sq_mapping,
                          )

    test_set = KTDataset(q_num, s_num, test_q, test_s, test_y, test_real_len, max_seq_len,
                         remapping=True,
                         qs_mapping=qs_mapping,
                         sq_mapping=sq_mapping,
                         )

    return train_set, test_set, train_set.get_qs_matrix()


# def load_assist09_q(args):
#     data_dir = 'Dataset/' + args.dataset
#     data = np.load(os.path.join(data_dir, args.dataset + '.npz'))
#     y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']
#
#     args.s_num, args.q_num = data['skill_num'], data['problem_num']
#
#     train_data, test_data = train_test_split([y, skill, problem, real_len])  # [y, skill, pro, real_len]
#     train_y, train_skill, train_problem, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
#     test_y, test_skill, test_problem, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
#     args.q_num = max(np.max(train_problem), np.max(test_problem))
#
#     train_set = KTDataset(args.q_num, args.s_num, train_problem, train_skill, train_y, train_real_len,
#                           max_seq_len=args.max_seq_len, remapping=False)
#     test_set = KTDataset(args.q_num, args.s_num, test_problem, test_skill, test_y, test_real_len,
#                          max_seq_len=args.max_seq_len, remapping=False)
#     # todo : load qs_matrix
#     qs_matrix = None
#     return train_set, test_set, qs_matrix


def load_assist17_s(args):
    print("Loading assist 2017 skill datasets")

    n_question = 102
    n_pid = 3162

    args.s_num = 102
    args.q_num = 3162

    data_path = os.path.join("Dataset", "../../Dataset/assist2017_pid", "assist2017_pid.csv")
    from KT.dataset_loader.assist_17 import PID_DATA
    dat = PID_DATA(n_question=n_question,
                   seqlen=args.max_seq_len, separate_char=',')

    q_data, qa_data, pid = dat.load_data(data_path)

    skill = q_data.astype(int)
    y = (qa_data > args.s_num * numpy.ones_like(qa_data)).astype(int)

    problem = pid.astype(int)
    real_len = (q_data > numpy.zeros_like(q_data)).sum(axis=1).astype(int)

    train_data, test_data = train_test_split([y, skill, problem, real_len])

    train_y, train_skill, train_problem, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
    test_y, test_skill, test_problem, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]
    args.q_num = max(np.max(train_problem), np.max(test_problem))
    assert args.q_num == 3162

    train_set = KTDataset_SA(args.q_num, args.s_num, train_problem, train_skill, train_y, train_real_len,
                             max_seq_len=args.max_seq_len)
    test_set = KTDataset_SA(args.q_num, args.s_num, test_problem, test_skill, test_y, test_real_len,
                            max_seq_len=args.max_seq_len)

    qs_matrix = None
    return train_set, test_set, qs_matrix


def load_buaa(args):
    print("Loading Beihang1819 Dataset")
    print(args)
    from Dataset.beihang.analyse import load, load_qmatrix
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
    return train_set, test_set, q_matrix


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


def load_dataset(args):
    if args.dataset == 'assist09-q':

        train_set, test_set, qs_matrix = load_assist09_q(args)
    elif args.dataset == 'assist09-s':
        train_set, test_set, qs_matrix = load_assist09_s(args)
    elif args.dataset == 'assist17-s':
        train_set, test_set, qs_matrix = load_assist17_s(args)
    elif args.dataset == 'beihang':
        train_set, test_set, qs_matrix = load_buaa(args)
    elif args.dataset == 'buaa18a':
        pass
    elif args.dataset == 'buaa18s':
        pass
    elif args.dataset == 'ednet-qs':
        from KT.dataset_loader.ednet import load_ednet_qs
        train_set, test_set, qs_matrix = load_ednet_qs(args)
    elif args.dataset == 'ednet-re':
        train_set, test_set, qs_matrix = load_ednet_re(args)
    elif args.dataset == 'ednet':
        from KT.dataset_loader.ednet import load_ednet
        train_set, test_set, qs_matrix = load_ednet(args)
    elif args.dataset == 'junyi':
        train_set, test_set, qs_matrix = load_junyi(args)
    elif args.dataset == 'junyi2':
        train_set, test_set, qs_matrix = load_junyi2(args)
    else:
        train_set = None
        test_set = None
        qs_matrix = None

    ratio = [0.7, 0.25, 0.05]
    # if dataset_info['s_graph'] != 'False':
    #     s_graph = np.load(dataset_info['s_graph'])
    if args.data_augment:
        print('Use Data Augmentation')
        train_set.augment()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=args.shuffle)
    '''
        Modification Required
        need a better way to load qs_matrix cuz the original file is like a crap of shit 
    '''
    return {'train_loader': train_loader, 'test_loader': test_loader, 'qs_matrix': qs_matrix}


def load_model(args):
    def load_checkpoint(model, optimizer, checkpoint_PATH):
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('Loading Checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
        return model, optimizer

    model = None
    optimizer = None

    if args.model == 'DKT':
        from KT.models.DKT import DKT
        model = DKT(feature_dim=2 * args.q_num + 1,
                    embed_dim=args.embed_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.q_num + 1)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == 'DKT_AUG':
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
    elif args.model == 'DKT_PEBG':
        from KT.models.DKT import DKT_PEBG

        model = DKT_PEBG(pro_num=args.q_num + 1,
                         skill_num=args.s_num,
                         hidden_dim=args.hidden_dim,
                         embed_dim=args.embed_dim,
                         output_dim=args.q_num
                         )
        if args.pretrain in ['load', 'scratch']:
            model.load_pretrain_embedding(args.pretrain_embed_file)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == 'DKT_PLUS':
        from KT.models.DKT import DKT_PLUS
        model = DKT_PLUS(
            q_num=args.q_num + 1,
            s_num=args.s_num,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            output_dim=args.q_num
        )
        model.set_qs_matrix(args.qs_matrix)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == 'SAKT':
        ## why this part is not updated in remote repo?
        from KT.models.SAKT import SAKT
        model = SAKT(q_num=args.q_num, seq_len=args.max_seq_len, embed_dim=args.embed_dim, heads=args.n_heads,
                     dropout=args.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.model == 'SAKT_SKILL':
        ## why this part is not updated in remote repo?

        from KT.models.SAKT import SAKT_SKILL
        model = SAKT_SKILL(q_num=args.q_num, seq_len=args.max_seq_len, embed_dim=args.embed_dim, heads=1,
                           dropout=args.dropout)
        model.set_qs_matrix(args.qs_matrix)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    elif args.model == 'DKVMN_RE':
        from KT.models.DKVMN_RE import DKVMN_RE
        q_embed_dim = args.embed_dim
        qa_embed_dim = args.embed_dim
        final_fc_dim = 50
        model = DKVMN_RE(n_question=args.q_num,
                         batch_size=args.batch_size,
                         q_embed_dim=q_embed_dim,
                         qa_embed_dim=qa_embed_dim,
                         memory_size=args.qs_matrix.shape[1],
                         memory_key_state_dim=q_embed_dim,
                         memory_value_state_dim=qa_embed_dim,
                         final_fc_dim=final_fc_dim)
        model.set_qs_matrix(args.qs_matrix)
        model.init_embeddings()
        model.init_params()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == 'AKT':
        from KT.models.AKT import AKT
        print(args.q_num)
        model = AKT(s_num=args.s_num, q_num=args.q_num, n_blocks=1, d_model=256,
                    dropout=0.05, kq_same=1, model_type='akt', l2=1e-5)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == 'SAINT':
        pass
        # TODO : Update SAINT Model
        # model = saint(dim_model=128,
        #               num_en=6,
        #               num_de=6,
        #               heads_en=8,
        #               heads_de=8,
        #               total_ex=total_ex,
        #               total_cat=total_cat,
        #               total_in=total_in,
        #               seq_len=args.seq_len
        #               )
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
        from KT.models.GKT import GKT
        graph_model = 'MHA'
        # graph = build_dense_graph(question_num)
        graph = None
        model = GKT(args.s_num, args.hidden_dim, args.embed_dim, args.edge_types, args.graph_type, graph=graph,
                    graph_model=None,
                    dropout=args.dropout, has_cuda=torch.cuda.is_available())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == 'QGKT':
        from KT.models.QGKT import QGKT

        def s_graph_gen(s_n):
            s_graph = np.ones((s_n, s_n))
            s_graph = s_graph / (s_n - 1)
            np.fill_diagonal(s_graph, 0)

            return s_graph

        model = QGKT(question_num=args.q_num, skill_num=args.s_num, hidden_dim=args.hidden_dim,
                     embedding_dim=args.embed_dim,
                     qs_matrix=args.qs_matrix, s_graph=s_graph_gen(args.s_num))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        pass
    if args.checkpoint_dir is not None and args.train_from_scratch is not True:
        from KT.util.checkpoint import CheckpointManager

        model, optimizer, args.current_epoch = CheckpointManager.load_checkpoint_by_hyperparameters(
            model=model,
            optimizer=optimizer,
            directory=args.checkpoint_dir,
            model_name=args.model,
            dataset=args.dataset,
            hyperparameters=model.get_hyperparameters()
        )

        assert model != "Failed to load"
        print(f"Successfully load checkpoint {args.checkpoint_dir} from epoch {args.current_epoch}")

    return model, optimizer


def observe_similar_question_improvement(args, target_q, target_s):
    pass


if __name__ == '__main__':
    load_assist09_s(None)
