import pandas as pd
import numpy as np
import os

def merge_two(f1,f2,out):
    s1 = open(f1,'r').readlines()
    s2 = open(f2,'r').readlines()

    for s in s2:
        s1.append(s)

    for idx,_ in enumerate(s1):
        if s1[idx][-1]!= '\n':
            s1[idx]=s1[idx]+'\n'
    f = open(out,'w')

    f.writelines(s1)


merge_two('2018_s/2018_s_train.csv','2018_s/2018_s_test.csv','2018_s/2018_s.csv')

def prob_map(file,index_file,target_path):

    index_file = pd.read_excel(index_file)
    #index_file = pd.read_csv(index_file)
    print(index_file)
    writer = open(target_path,'w')
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            idx = i
            line = lines[i]
            print(line)
            if line[-1]=='\n' : line = line[:-1]
            if idx % 3 == 0:
                writer.write(line+'\n')
                print(line+'\n')
            elif idx % 3 == 1:
                question_seq = line.split(',')
                question_seq = [int(s) for s in question_seq]
                new_seq = []
                for q in question_seq:
                    q= int(q)
                    row = index_file.loc[index_file['index'] == q]

                    q_remap = int(row['id'])

                    new_seq.append(str(q_remap))
                writer.write(','.join(new_seq)+'\n')
                print(','.join(new_seq)+'\n')
            else:
                print(line)
                writer.write(line+'\n')

    f.close()



prob_map('2018_s/2018_s.csv', '2018_s/2018_s_exercise.xlsx', '2018_s/2018_s_map.txt')


