import numpy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import torch.nn.functional as F
import numpy as np

random_seed = 42


def build_dense_graph(node_num):
    print(node_num)
    graph = 1. / (node_num - 1) * np.ones((node_num, node_num))
    np.fill_diagonal(graph, 0)
    graph = torch.from_numpy(graph).float()
    return graph


class KTDataset(Dataset):
    def __init__(self, q_num, s_num, questions, skills, answers, seq_len, max_seq_len, qs_mapping=None, sq_mapping=None,
                 remapping=False, item_start_from_one=True):

        super(KTDataset, self).__init__()
        self.q_num = q_num
        self.s_num = s_num
        self.questions = questions
        self.skills = skills

        if skills is None:
            self.skills = self.questions
        self.answers = answers
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        # need some calculation

        self.qs_mapping = qs_mapping
        self.sq_mapping = sq_mapping
        self.q_trans_graph = None
        self.s_trans_graph = None
        if remapping:
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

            self.sid_remapping_reverse = {value: key for key, value in self.sid_remapping.items()}
            self.qid_remapping_reverse = {value: key for key, value in self.qid_remapping.items()}

            qid_vectorized_mapping = np.vectorize(lambda x: 0 if x < 0 else self.qid_remapping[x])
            sid_vectorized_mapping = np.vectorize(lambda x: 0 if x < 0 else self.sid_remapping[x])

            self.questions = qid_vectorized_mapping(self.questions)
            if self.skills is not None:
                self.skills = sid_vectorized_mapping(self.skills)

            # re-calculate qs-matrix base on remapped index
            self.qs_matrix = np.ndarray([self.q_num + 1, self.s_num + 1], dtype=int)
            for q,s_list in qs_mapping.items():
                qid_remapped = self.qid_remapping[q]
                s_list = [self.sid_remapping[s] for s in s_list]
                for sid_remapped in s_list:
                    self.qs_matrix[qid_remapped,sid_remapped] = 1

        answers_one = (answers == 1.0)

        assert questions.shape == answers.shape
        self.features = answers_one * self.q_num + self.questions

        assert np.max(self.answers) <=1
        assert np.min(self.answers) >=-1
        # this loading process is way fucking too slow that can be optimized greatly


    def init_by_raw_sequences(self, q_seq, a_seq, qs_mapping):
        pass

    def __getitem__(self, index):
        '''
            this part requires a lot of update to fit different datasets
        '''

        return self.features[index], self.questions[index], self.skills[index], self.answers[index], self.seq_len[index]

    def __len__(self):
        return len(self.features)

    def set_qs_mapping(self, qs_mapping):
        self.qs_mapping = qs_mapping

    def set_sq_mapping(self, sq_mapping):
        self.sq_mapping = sq_mapping

    def get_qs_matrix(self):
        return self.qs_matrix

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

        self.features = np.array(self.features)
        self.questions = np.array(self.questions)
        self.skills = np.array(self.skills)
        self.answers = np.array(self.answers)
        self.seq_len = np.array(self.seq_len)

    def purify(self, min_len=10):
        pass

    def print(self):
        print(self.features)
        print(self.features.shape)
        print(self.questions.shape)
        print(self.answers.shape)
        print(self.skills.shape)
        print(self.seq_len.shape)

    def gen_skill_trans_graph(self):
        pass

    def set_question_trans_graph(self,trans_matrix:torch.FloatTensor):
        assert trans_matrix.shape[0] == trans_matrix.shape[1]
        assert trans_matrix.shape[0] == self.q_num
        self.q_trans_graph = trans_matrix

    def get_question_trans_graph(self):
        '''
            Is it a valid approach to obtain trans-graph on both train and test set?
        '''
        if self.q_trans_graph is not None:
            return self.q_trans_graph

        self.q_trans_graph = torch.zeros([self.q_num,self.q_num])
        data_num,seq_len = self.questions.shape

        for i in range(data_num):
            for j in range(seq_len-1):
                qi = self.questions[i, j].int()
                qj = self.questions[i][j + 1].int()
                if qi == 0 or qj == 0:
                    continue

                self.q_trans_graph[qi - 1, qj - 1] += 1

        row_sums = self.q_trans_graph.sum(dim = 1,keepdim=True)
        self.q_trans_graph = self.q_trans_graph / row_sums.unsqueeze(-1)

        return self.q_trans_graph

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


import numpy as np


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


def test_get_question_trans_graph():
    questions = torch.FloatTensor([
        [1, 2, 3],
        [1, 2, 1],
        [2, 3, 1]
    ])
    target = torch.FloatTensor([
        [1, 2, 3],
        [1, 3, 1],
        [2, 3, 1]
    ])
    q_num = 3
    q_trans_graph = torch.zeros([q_num, q_num])
    data_num, seq_len = questions.shape

    for i in range(data_num):
        for j in range(seq_len - 1):
            qi = questions[i, j].int()
            qj = questions[i][j+1].int()
            if qi == 0 or qj == 0:
                continue


            q_trans_graph[qi-1, qj-1] += 1

    row_sums = q_trans_graph.sum(dim=1)
    print(q_trans_graph)
    print(row_sums)
    print(row_sums.unsqueeze(-1))
    q_trans_graph = q_trans_graph / row_sums.unsqueeze(-1)

    print(q_trans_graph)

    assert q_trans_graph == target
if __name__ == '__main__':
    test_get_question_trans_graph()
