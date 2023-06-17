import pandas as pd
import numpy as np

df = pd.read_csv('Dataset/beihang/q_matrix.csv')
def remap(q):
    return df.loc[df['id']==q].index.tolist()[0]+1



def load_buaa(data_path,max_seq_len=200):
    seq_len_list = []
    question_list = []
    answer_list = []
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
                # remap question_seq here

                question_seq = [remap(q) for q in question_seq]
                for q in question_seq:
                    problem_set.add(q)



                padding_len = max_seq_len-len(question_seq)
                while padding_len>0:
                    question_seq.append(0)
                    padding_len-=1
                question_list.append(question_seq)
            else:
                answer_seq = line.split(',')
                answer_seq = [int(s) for s in answer_seq][:max_seq_len]

                padding_len = max_seq_len-len(answer_seq)
                while padding_len>0:
                    answer_seq.append(-1)
                    padding_len-=1
                answer_list.append(answer_seq)
                # feature_list.append(
                #     [2 * (q-1) + answer_seq[idx]  for idx, q in enumerate(skill_seq)]
                # )
                #feature_list.append([answer_seq[idx] * question_num + q for idx, q in enumerate(question_seq)])

    f.close()
    Q = np.array(question_list)
    #print(Q.shape)
    A = np.array(answer_list)
    #print(A.shape)
    L = np.array(seq_len_list)
    #print(L.shape)
    return Q, A,L


def reload_all():
    Q1, A1, L1 = load_buaa('beihang/2018_s_map.txt')
    Q2, A2, L2 = load_buaa('beihang/2018_w_map.txt')
    Q3, A3, L3 = load_buaa('beihang/2019_s_map.txt')
    Q4, A4, L4 = load_buaa('beihang/2019_w_map.txt')

    Q = np.concatenate([Q1, Q2, Q3, Q4])
    A = np.concatenate([A1, A2, A3, A4]).astype(np.float32)
    L = np.concatenate([L1, L2, L3, L4])
    # Q = np.concatenate([Q1])
    # A = np.concatenate([A1]).astype(np.float32)
    # L = np.concatenate([L1])

    np.savez('beihang/BUAA.npz',Q=Q,A=A,L=L)

    return  Q,A,L

def load():
    buaa_dict = np.load('Dataset/beihang/BUAA.npz')
    return buaa_dict['Q'],buaa_dict['A'],buaa_dict['L']

def load_qmatrix():

    df = pd.read_csv('Dataset/beihang/q_matrix.csv')
    q_matrix = df.to_numpy()[:,2:]
    return q_matrix