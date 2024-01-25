import logging

import numpy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import torch.nn.functional as F
import numpy as np
from KT.util.decorators import timing_decorator

random_seed = 42


def convert_to_tensor(arr):
    """
    Convert NumPy array to PyTorch tensor with the original datatype.

    Parameters:
    - arr: NumPy array

    Returns:
    - torch.Tensor: PyTorch tensor with the same datatype as the original array
    """
    if isinstance(arr, np.ndarray):
        # Get the original datatype of the NumPy array
        dtype = arr.dtype

        # Convert the NumPy array to a PyTorch tensor with the original datatype
        torch_tensor = torch.from_numpy(arr)

        return torch_tensor
    elif isinstance(arr, torch.Tensor):
        return arr
    else:
        raise ValueError("Input is not a NumPy array")


class KTDataset(Dataset):
    def __init__(self, q_num, s_num, questions, skills, answers, seq_len, max_seq_len, qs_mapping=None, sq_mapping=None,
                 remapping=False, item_start_from_one=True):

        super(KTDataset, self).__init__()
        self.Q_PADDING = 0
        self.S_PADDING = 0
        self.A_PADDING = -1
        self.I_PADDING = 0

        self.q_num = q_num
        self.s_num = s_num
        self.questions = convert_to_tensor(questions)
        self.skills = convert_to_tensor(skills)

        # if skills is None:
        #     self.skills = self.questions
        self.answers = convert_to_tensor(answers)
        self.seq_len = convert_to_tensor(seq_len)

        self.max_seq_len = max_seq_len
        # need some calculation

        assert torch.max(self.seq_len) <= self.max_seq_len

        self.qs_mapping = qs_mapping
        self.sq_mapping = sq_mapping
        self.q_trans_graph = None
        self.s_trans_graph = None

        self.return_all_length_seq = False

        if remapping:
            print('remapping the qid and sid to [1,q_num] and [1,s_num]')
            '''
                remapping the qid and sid to [1,q_num] and [1,s_num]
                while doing this, the qs mapping also have to be updated
            '''
            self.sid_remapping = {}
            self.qid_remapping = {}
            for i in range(1, len(self.qs_mapping) + 1):
                self.qid_remapping[list(self.qs_mapping.keys())[i - 1]] = i
            for i in range(1, len(self.sq_mapping) + 1):
                self.sid_remapping[list(self.sq_mapping.keys())[i - 1]] = i

            # assert 16229 in self.sq_mapping.keys()
            import pickle
            with open('ednet-remapping.pkl', 'wb') as f:
                pickle.dump((self.sid_remapping, self.qid_remapping), f)
            self.sid_remapping_reverse = {value: key for key, value in self.sid_remapping.items()}
            self.qid_remapping_reverse = {value: key for key, value in self.qid_remapping.items()}

            qid_vectorized_mapping = np.vectorize(lambda x: 0 if x < 0 else self.qid_remapping[x])

            self.questions = convert_to_tensor(qid_vectorized_mapping(self.questions))
            if self.skills is not None:
                sid_vectorized_mapping = np.vectorize(lambda x: 0 if x < 0 else self.sid_remapping[x])
                self.skills = convert_to_tensor(sid_vectorized_mapping(self.skills))

            # re-calculate qs-matrix base on remapped index
            self.qs_matrix = np.zeros([self.q_num + 1, self.s_num + 1], dtype=int)
            for q, s_list in qs_mapping.items():
                qid_remapped = self.qid_remapping[q]
                s_list = [self.sid_remapping[s] for s in s_list]
                for sid_remapped in s_list:
                    self.qs_matrix[qid_remapped, sid_remapped] = 1

        answers_one = (self.answers == 1.0)
        self.qs_matrix = torch.from_numpy(self.qs_matrix)
        assert questions.shape == answers.shape
        self.features = answers_one * self.q_num + self.questions
        assert isinstance(self.features, torch.Tensor)
        assert torch.max(self.answers) <= 1
        assert torch.min(self.answers) >= -1
        # this loading process is way fucking too slow that can be optimized greatly

    def init_by_raw_sequences(self, q_seq, a_seq, qs_mapping):
        pass

    def __getitem__(self, index):
        '''
            this part requires a lot of update to fit different datasets
        '''

        if self.return_all_length_seq:
            return self.s_feat[index], self.s_question[index], self.s_skill[index], self.s_answer[index], \
                   self.s_seq_len[index]
        return self.features[index], self.questions[index], self.skills[index], self.answers[index], self.seq_len[index]

    def __len__(self):
        if self.return_all_length_seq:
            return len(self.s_feat)
        return len(self.features)

    def save(self, datapath):
        data_dict = {
            'features': self.features,
            'questions': self.questions,
            'skills': self.skills,
            'answers': self.answers,
            'seq_len': self.seq_len if self.return_all_length_seq else None,
            's_feat': self.s_feat,
            's_question': self.s_question,
            's_skill': self.s_skill,
            's_answer': self.s_answer,
            's_seq_len': self.s_seq_len,
            'return_all_length_seq': self.return_all_length_seq
        }
        torch.save(data_dict, datapath)

    def load(self, datapath):
        data_dict = torch.load(datapath)
        return_all_length_seq = data_dict.get('return_all_length_seq', False)

        self.features = data_dict['features'],
        self.questions = data_dict['questions'],
        self.skills = data_dict['skills'],
        self.answers = data_dict['answers'],
        self.seq_len = data_dict['seq_len'],
        self.return_all_length_seq = return_all_length_seq

        if return_all_length_seq:
            self.s_feat = data_dict['s_feat'],
            self.s_question = data_dict['s_question'],
            self.s_skill = data_dict['s_skill'],
            self.s_answer = data_dict['s_answer'],
            self.s_seq_len = data_dict['s_seq_len'],
            self.return_all_length_seq = return_all_length_seq

    def merge(self, another_kt_set):
        if not isinstance(another_kt_set, KTDataset):
            raise ValueError("Can only merge with datasets of the same class (MyDataClass)")

        self.seq_len = torch.cat([self.seq_len, another_kt_set.seq_len])
        self.features = torch.vstack([self.features, another_kt_set.features])
        self.questions = torch.vstack([self.questions, another_kt_set.questions])
        self.skills = torch.vstack([self.skills, another_kt_set.skills])
        self.answers = torch.vstack([self.answers, another_kt_set.answers])

        if hasattr(self, 's_seq_len'):
            self.s_seq_len = torch.cat([self.s_seq_len, another_kt_set.s_seq_len])
            self.s_feat = torch.vstack([self.s_feat, another_kt_set.s_feat])
            self.s_skill = torch.vstack([self.s_skill, another_kt_set.s_skill])
            self.s_question = torch.vstack([self.s_question, another_kt_set.s_question])
            self.s_answer = torch.vstack([self.s_answer, another_kt_set.s_answer])

        return self

    def set_qs_mapping(self, qs_mapping):
        self.qs_mapping = qs_mapping

    def set_sq_mapping(self, sq_mapping):
        self.sq_mapping = sq_mapping

    def get_qs_matrix(self):
        return self.qs_matrix

    def q2s(self, qid):
        row = self.qs_matrix[qid, :]
        skills = [sid for sid, value in enumerate(row) if value > 0]
        return skills

    def s2q(self, sid):
        col = self.qs_matrix[:, sid]
        questions = [qid for qid, value in enumerate(col) if value > 0]
        return questions

    def analyse(self):
        q_num = 0
        for seq in self.questions:
            for q in seq:
                q_num = max(q, q_num)
        print('Dataset Question Number : ', q_num)
        freq_list = [0 for i in range(q_num + 1)]
        right_list = [0 for i in range(q_num + 1)]
        wrong_list = [0 for i in range(q_num + 1)]
        for i, seq in enumerate(self.questions):
            for j, q in enumerate(seq):
                freq_list[q] += 1
                if self.answers[i][j] == 0:
                    wrong_list[q] += 1
                else:
                    right_list[q] += 1
        # import matplotlib.pyplot as plt
        #
        # plt.subplot(2,1,1)
        # for i in range(len(freq_list)):
        #     plt.bar(i,freq_list[i])
        # plt.subplot(2,1,2)
        # diff_list = [wrong_list[idx]/(0.00001+right_list[idx]+wrong_list[idx]) for idx,_ in enumerate(right_list)]
        # for i in range(len(diff_list)):
        #     plt.bar(i,diff_list[i])
        # plt.show()

    def augment(self):
        import numpy as np
        self.features = self.features.tolist()
        self.questions = self.questions.tolist()
        self.skills = self.skills.tolist()
        self.answers = self.answers.tolist()
        self.seq_len = self.seq_len.tolist()
        total_len = len(self.features)
        max_l = self.max_seq_len

        def pad(target, value, max_len):

            pad_len = max_len - len(target)
            target = target + [value for i in range(pad_len)]
            return target

        for i in range(0, total_len):

            seq_len = self.seq_len[i]
            if seq_len < 5: continue
            import random

            mid = random.randint(int(seq_len / 3), int(seq_len / 3 * 2))
            assert mid >= 0
            assert mid <= seq_len

            features1 = pad(self.features[i][:mid], 0, max_l)
            features2 = pad(self.features[i][mid:], 0, max_l)
            self.features.append(features1)
            self.features.append(features2)

            questions1 = pad(self.questions[i][:mid], 0, max_l)
            questions2 = pad(self.questions[i][mid:], 0, max_l)
            self.questions.append(questions1)
            self.questions.append(questions2)

            skills1 = pad(self.skills[i][:mid], 0, max_l)
            skills2 = pad(self.skills[i][mid:], 0, max_l)
            self.skills.append(skills1)
            self.skills.append(skills2)

            answers1 = pad(self.answers[i][:mid], -1, max_l)
            answers2 = pad(self.answers[i][mid:], -1, max_l)
            self.answers.append(answers1)
            self.answers.append(answers2)

            l1 = len(features1)
            l2 = len(features2)
            self.seq_len.append(l1)
            self.seq_len.append(l2)

        self.features = torch.Tensor(self.features)
        self.questions = torch.Tensor(self.questions)
        self.skills = torch.Tensor(self.skills)
        self.answers = torch.Tensor(self.answers)
        self.seq_len = torch.Tensor(self.seq_len)

    def purify(self, min_len=10):
        pass

    def print(self):
        print(self.features)
        print(self.features.shape)
        print(self.questions.shape)
        print(self.answers.shape)
        print(self.skills.shape)
        print(self.seq_len.shape)

    @timing_decorator
    def get_skill_trans_graph(self):
        '''
            Is it a valid approach to obtain trans-graph on both train and test set?
        '''
        if self.s_trans_graph is not None:
            return self.s_trans_graph

        self.s_trans_graph = torch.zeros([self.s_num, self.s_num]).to('cpu')
        data_num, seq_len = self.questions.shape
        q2s = [None for i in range(self.qs_matrix.shape[0])]
        for i in range(data_num):
            for j in range(seq_len - 1):
                print(i, j)
                qi = self.questions[i, j]
                qj = self.questions[i][j + 1]
                if qi == 0 or qj == 0:
                    continue

                def look_up_or_cache(qx):
                    if q2s[qx] is None:
                        q2s[qx] = torch.nonzero(self.qs_matrix[qx,] > 0)
                    return q2s[qx]

                related_skills_qi = look_up_or_cache(qi)
                related_skills_qj = look_up_or_cache(qj)

                for _si in related_skills_qi:
                    for _sj in related_skills_qj:
                        si = _si.item()
                        sj = _sj.item()
                        # si_original = self.sid_remapping_reverse[si]
                        # qi_original = self.qid_remapping_reverse[qi]

                        # assert si_original in self.qs_mapping[qi_original]
                        self.s_trans_graph[si - 1, sj - 1] += 1

        row_sums = self.s_trans_graph.sum(dim=1, keepdim=False)
        print(self.s_trans_graph.shape)
        self.s_trans_graph = self.s_trans_graph / row_sums.unsqueeze(-1)
        np.save('s_trans_matrix_' + str(data_num) + '.npy', self.s_trans_graph.numpy())
        return self.s_trans_graph

    def set_question_trans_graph(self, trans_matrix: torch.FloatTensor):
        assert trans_matrix.shape[0] == trans_matrix.shape[1]
        assert trans_matrix.shape[0] == self.q_num
        self.q_trans_graph = trans_matrix

    @timing_decorator
    def get_question_trans_graph(self):
        '''
            Is it a valid approach to obtain trans-graph on both train and test set?
        '''
        if self.q_trans_graph is not None:
            return self.q_trans_graph

        self.q_trans_graph = torch.zeros([self.q_num, self.q_num]).to('cpu')
        data_num, seq_len = self.questions.shape

        for i in range(data_num):
            for j in range(seq_len - 1):
                qi = self.questions[i, j]
                qj = self.questions[i][j + 1]
                if qi == 0 or qj == 0:
                    continue

                self.q_trans_graph[qi - 1, qj - 1] += 1

        row_sums = self.q_trans_graph.sum(dim=1, keepdim=False)
        print(self.q_trans_graph.shape)
        self.q_trans_graph = self.q_trans_graph / row_sums.unsqueeze(-1)

        return self.q_trans_graph

    @timing_decorator
    def generate_seq_all_length(self):
        from copy import deepcopy
        self.s_feat = None
        self.s_question = None
        self.s_seq_len = []
        self.s_skill = None
        self.s_answer = None

        def append_row(array: np.ndarray, row: np.ndarray):
            if array is None:
                return row
            return np.vstack((array, row))

        def generate_all_len_array(input_array, seq_len, pad_val):

            result_array = pad_val * np.ones_like(input_array)

            result_array = np.tile(result_array, (seq_len, 1))

            for i in range(seq_len):
                result_array[i, :i + 1] = input_array[:i + 1]

            return result_array[1:, :]

        for idx, (feat, question, skill, answer, seq_len) in enumerate(self):

            if seq_len < 2:
                continue

            self.s_feat = append_row(self.s_feat, generate_all_len_array(feat, seq_len, self.I_PADDING))
            self.s_question = append_row(self.s_question, generate_all_len_array(question, seq_len, self.Q_PADDING))
            self.s_answer = append_row(self.s_answer, generate_all_len_array(answer, seq_len, self.A_PADDING))
            self.s_skill = append_row(self.s_skill, generate_all_len_array(skill, seq_len, self.S_PADDING))
            self.s_seq_len.extend([i for i in range(2, seq_len + 1)])
            assert self.s_feat.shape[0] == len(self.s_seq_len)

        self.s_seq_len = torch.IntTensor(self.s_seq_len)

        self.return_all_length_seq = True

    @timing_decorator
    def generate_multi_skill_seq(self):
        """
            make sure this function is used on question-level sequence, that the multi-skill interaction
            is NOT split into multi interaction
            should we just do it in preprocessing

        """

        interaction_s = []
        question_s = []
        answer_s = []
        skill_s = []
        seq_len_s = []
        idx = 0

        for idx, (feat, question, skill, answer, seq_len) in enumerate(self):
            print(idx, len(self))
            qt = question[seq_len - 1]
            at = answer[seq_len - 1]
            if seq_len < self.max_seq_len:
                assert question[seq_len] == self.Q_PADDING
            q_seq_old = question[:seq_len - 1]
            a_seq_old = answer[:seq_len - 1]
            q_seq_old_duplicate = []
            a_seq_old_duplicate = []
            s_seq_old = []
            for i in range(len(q_seq_old)):
                qi = q_seq_old[i]
                ai = a_seq_old[i]
                for s in self.q2s(qi):
                    q_seq_old_duplicate.append(qi)
                    a_seq_old_duplicate.append(ai)
                    s_seq_old.append(s)
            for s in self.q2s(qt):
                question_s.append(q_seq_old_duplicate + [qt])
                answer_s.append(a_seq_old_duplicate + [at])
                skill_s.append(s_seq_old + [s])

        '''
            base on new q_seq, a_seq, s_seq, re-generate the features tensors
        '''

        def pad_or_cut(seq, pad_value, max_seq_len):
            if len(seq) > max_seq_len:
                return seq[:max_seq_len]
            return seq + [pad_value for i in range(max_seq_len - len(seq))]

        for i in range(len(question_s)):
            seq_len_s.append(min(self.max_seq_len, len(question_s[i])))
            question_s[i] = pad_or_cut(question_s[i], self.Q_PADDING, self.max_seq_len)
            skill_s[i] = pad_or_cut(skill_s[i], self.S_PADDING, self.max_seq_len)
            answer_s[i] = pad_or_cut(answer_s[i], self.A_PADDING, self.max_seq_len)

        self.s_seq_len = torch.IntTensor(seq_len_s)
        self.s_question = torch.IntTensor(question_s)
        self.s_skill = torch.IntTensor(skill_s)
        self.s_answer = torch.IntTensor(answer_s)
        self.s_feat = self.s_question + self.s_answer * self.q_num
        self.s_feat[self.s_question == self.Q_PADDING] = self.I_PADDING

        self.return_all_length_seq = True


class KTDataset_SA(KTDataset):
    def __init__(self, q_num, s_num, questions, skills, answers, seq_len, max_seq_len):
        super(KTDataset_SA, self).__init__(q_num, s_num, questions, skills, answers, seq_len, max_seq_len)
        answers_one = (answers == 1.0)
        self.features = answers_one * self.s_num + self.skills


def pad_collate(batch):
    (features, questions, answers) = zip(*batch)
    features = [torch.LongTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.LongTensor(ans) for ans in answers]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    return feature_pad, question_pad, answer_pad


def load_KTData(data_path, question_num, max_seq_len, ratio, batch_size=50, data_shuffle=True):
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
            # assert f_seq[j] == a_seq[j] *question_num + q_seq[j]
            f_seq[j] = a_seq[j] * question_num + q_seq[j]
    kt_dataset = KTDataset(feature_list, question_list, answer_list)
    kt_dataset.analyse()

    dataset_size = len(kt_dataset)
    print(f'Dataset Size: {dataset_size}')

    train_size = int(ratio[0] * dataset_size)
    test_size = dataset_size - train_size

    train_set, val_set, test_set = random_split(kt_dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=data_shuffle, collate_fn=pad_collate)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=data_shuffle, collate_fn=pad_collate)

    print(train_loader)
    print(test_loader)
    return train_loader, test_loader


def is_ascending(seq):
    if len(seq) == 1: return False
    for i in range(0, len(seq) - 1):
        if seq[i] > seq[i - 1]: return False
    return True


def preprocess():
    csv_path = "/Users/watsonyang/PycharmProjects/MyKT/Dataset/assist2009_updated/assist2009_updated_test.csv"
    concept_num = 110
    seq_len_list = []
    question_list = []
    answer_list = []
    feature_list = []
    test_seq = [1, 2, 3, 4, 2, 1]
    print(is_ascending(test_seq))
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        print("Totol Sequence Number:", len(lines))
        ascending_count = 0
        for idx, line in enumerate(lines):
            if idx % 3 == 0:
                pass
            elif idx % 3 == 1:
                skill_seq = line.split(',')
                skill_seq = [int(s) for s in skill_seq]
                if is_ascending(skill_seq): ascending_count += 1

            else:
                pass
        print("Ascending Sequence Count:", ascending_count)


def wtf_get_question_trans_graph():
    # questions = torch.FloatTensor([
    #     [1, 2, 3],
    #     [1, 2, 1],
    #     [2, 3, 1]
    # ])
    q_num = 30000
    bs = 1000
    seqlen = 1000
    questions = torch.randint(0, q_num, [bs, seqlen])
    target = torch.FloatTensor([
        [1, 2, 3],
        [1, 3, 1],
        [2, 3, 1]
    ])
    q_trans_graph = torch.zeros([q_num, q_num])

    data_num, seq_len = questions.shape

    for i in range(data_num):
        for j in range(seq_len - 1):
            qi = questions[i, j].int()
            qj = questions[i][j + 1].int()
            if qi == 0 or qj == 0:
                continue

            q_trans_graph[qi - 1, qj - 1] += 1
    print(torch.sum(q_trans_graph))
    row_sums = q_trans_graph.sum(dim=1)
    row_sums[row_sums == 0] = 1e-5
    print(questions)
    print(q_trans_graph)
    print(row_sums)
    print(row_sums.unsqueeze(-1).shape)
    q_trans_graph = q_trans_graph / row_sums.unsqueeze(-1)

    print(q_trans_graph)

    from KT.util.visual import plot_heatmap
    plot_heatmap(q_trans_graph)
    #
    # assert q_trans_graph == target


def verify_dummy_dataset():
    q_num = 100
    s_num = 10

    markov_transition_q = torch.rand([q_num, q_num])

    def random_split_stick(total_length, n):
        # 生成 n-1 个随机数作为分割点
        split_points = np.sort(np.random.uniform(0, total_length, n - 1))

        # 计算每一份的长度
        lengths = np.zeros(n)
        lengths[0] = split_points[0]
        lengths[-1] = total_length - split_points[-1]
        for i in range(1, n - 1):
            lengths[i] = split_points[i] - split_points[i - 1]

        return lengths

    for _ in range(q_num):
        prob_i = random_split_stick(1, q_num)
        prob_i = prob_i / np.sum(prob_i)
        markov_transition_q[_, :] = torch.tensor(prob_i)

    @timing_decorator
    def generate_sequence(transition_matrix, initial_state, sequence_length):
        current_state = initial_state
        sequence = [current_state]

        for _ in range(sequence_length - 1):
            # 根据概率转移矩阵选择下一个状态

            # print(f'transition for state {_}',transition_matrix[current_state])
            wtf = transition_matrix[current_state]
            wtf = wtf.numpy()
            wtf = wtf / np.sum(wtf)
            next_state = np.random.choice(len(transition_matrix), p=wtf)
            sequence.append(next_state)
            current_state = next_state

        return sequence

    data_num = 200
    max_seq_len = 200
    markov_seq = generate_sequence(markov_transition_q, initial_state=1, sequence_length=2 * data_num * max_seq_len)

    print(markov_seq)

    questions = torch.ones([data_num, max_seq_len]) * -1
    for i in range(data_num):
        for j in range(max_seq_len):
            k = i * max_seq_len + j
            questions[i, j] = markov_seq[k]
    questions = questions.int()
    print(questions)
    skills = questions
    qs_mapping = {_: set() for _ in range(q_num)}
    sq_mapping = {_: set() for _ in range(s_num)}

    # for i in range(0,s_num):
    #     for j in range(0,q_num):
    #
    #         qs_mapping[j].add(i)
    #         sq_mapping[i].add(j)
    for j in range(0, q_num):
        qs_mapping[j].add(j % 10)
        sq_mapping[j % 10].add(j)

    answers = torch.ones_like(questions).int()
    seq_len = torch.IntTensor([max_seq_len, max_seq_len])
    print(qs_mapping)
    print(sq_mapping)
    dummyset = KTDataset(q_num, s_num, questions, None, answers, seq_len, max_seq_len, qs_mapping=qs_mapping,
                         sq_mapping=sq_mapping,
                         remapping=True, item_start_from_one=True)

    q_trans = dummyset.get_question_trans_graph()
    s_trans = dummyset.get_skill_trans_graph()
    print(markov_transition_q)
    print(q_trans)
    print(s_trans)

    from KT.util.visual import plot_multiple_heatmap
    plot_multiple_heatmap([markov_transition_q[:10, :10], q_trans[:10, :10], s_trans])
    return dummyset


def verify_generate_multi_skill_sequence():
    from KT.util.args import set_parser
    args = set_parser().parse_args()
    from KT.util.kt_util import load_dummy_dataset

    trainset, testset, qs_matrix = load_dummy_dataset(args)

    testset.generate_seq_all_length()
    testset.generate_multi_skill_seq()

    #
    for f, q, s, a, l in testset:
        print('--------------------')
        print(f[:l])
        print(q[:l])
        print(s[:l])
        print(a[:l])
        print(l)


if __name__ == '__main__':
    verify_generate_multi_skill_sequence()
    # wtf_get_question_trans_graph()
