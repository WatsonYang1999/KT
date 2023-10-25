# Code reused from https://github.com/jennyzhang0215/DKVMN.git
import os.path

import numpy as np
import math


class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen
    # data format
    # id, true_student_id
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        qa_data = []
        idx_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 0:
                student_id = lineID//3
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len(Q[len(Q)-1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            # int(A[i]) is in {0,1}
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    #print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    idx_data.append(student_id)
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_dataArray, qa_dataArray, np.asarray(idx_data)


class PID_DATA(object):
    def __init__(self, n_question,  seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_question = n_question
    # data format
    # id, true_student_id
    # pid1, pid2, ...
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        qa_data = []
        p_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 4 == 0:
                student_id = lineID//4
            if lineID % 4 == 2:
                Q = line.split(self.separate_char)
                if len(Q[len(Q)-1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            if lineID % 4 == 1:
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]

            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    problem_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            problem_sequence.append(int(P[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    p_data.append(problem_sequence)

        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat
        return q_dataArray, qa_dataArray, p_dataArray

if __name__ == '__main__':
    dataset = 'assist2017_pid'
    if dataset in {"assist2009_pid"}:
        n_question = 110
        batch_size = 64
        seqlen = 200

        data_name = dataset
        n_pid = 16891

    if dataset in {"assist2017_pid"}:
        batch_size = 64
        seqlen = 200

        data_name = dataset
        n_question = 102
        n_pid = 3162

    if dataset in {"assist2015"}:
        n_question = 100
        batch_size = 64
        seqlen = 200

        data_name = dataset

    if dataset in {"statics"}:
        n_question = 1223
        batch_size = 64
        seqlen = 200

        data_name = dataset


    if "pid" not in data_name:
        dat = DATA(n_question=n_question,
                   seqlen=seqlen, separate_char=',')
    else:
        dat = PID_DATA(n_question=n_question,
                       seqlen=seqlen, separate_char=',')
    os.chdir('C:\\Users\\12574\\Desktop\\KT-Refine')
    data_path = os.path.join("Dataset","assist2017_pid","assist2017_pid.csv")

    q_data, qa_data, pid = dat.load_data(data_path)

    # assert that in all sequences, the mapping from pid to q is always n to 1

    print(q_data.shape)
    print(pid.shape)

    s2q = {}
    q2s = {}
    for i in range(0, q_data.shape[0]):
        for j in range(0,q_data.shape[1]):
            s = int(q_data[i,j])
            q = int(pid[i,j])
            if q in q2s.keys():
                q2s[q].add(s)
            else:
                q2s[q] = set()
                q2s[q].add(s)

            if s in s2q.keys():
                s2q[s].add(q)
            else:
                s2q[s] = set()
                s2q[s].add(q)


    for k in q2s.keys():

        if not len(q2s[k]) == 1:
            print(q2s[k])
            print(k)

    for k in s2q.keys():

        if not len(s2q[k]) == 1:
            print(s2q[k])
            print(k)