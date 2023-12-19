import pandas as pd


def pad(target, value, max_len):
    for idx, pad_seq in enumerate(target):
        if len(target[idx]) > max_len:
            target[idx] = target[idx][:max_len]
        else:
            pad_len = max_len - len(pad_seq)
            target[idx] = pad_seq + [value for i in range(pad_len)]


def observe_single_question_improvement(target_q):
    seq_len = 20
    q_seq = [target_q for i in range(0, seq_len)]
    import random
    split_len = random.randint(0, seq_len)
    a_seq = [0 for i in range(0, split_len)] + [1 for i in range(0, split_len)]
    s_seq = [0]

    return [q_seq], [s_seq], [a_seq]


def create_custom_dataset():
    # q_seq, s_seq, a_seq = observe_single_question_improvement(2)
    q_seq = [
        [2,2],
        [2,2]
    ]

    a_seq = [
        [1,0],
        [1,1]
    ]
    columns = ['QuestionId', 'UserId', 'AnswerId', 'IsCorrect', 'CorrectAnswer', 'AnswerValue']
    df = pd.DataFrame(columns=columns)

    user_index = 1
    for i in range(0, len(q_seq)):
        for j in range(len(q_seq[i])):
            q = q_seq[i][j]

            a = a_seq[i][j]
            new_row = {'QuestionId': q, 'UserId': user_index, 'AnswerId': 0, 'IsCorrect': a, 'CorrectAnswer': 0,
                       'AnswerValue': 0}
            df = df.append(new_row, ignore_index=True)
        user_index += 1
    return df


if __name__ == '__main__':
    df = create_custom_dataset()
    print(df)
    df.to_csv('C:\\Users\\12574\\Desktop\\KT-Refine\\Dataset\\ednet-re\\data\\custom_data\\custom_task_1.csv')
