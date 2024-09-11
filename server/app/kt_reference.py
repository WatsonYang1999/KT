###
import math



def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def infer(records, next_question_id):
    '''
    :param records: [(question_id,score)]
    :return:
    '''

    import random
    # 生成一个介于 0 和 3 之间的随机浮点数
    max_q_id = -1
    total_score = 0
    for r in records:
        if max_q_id < r[0]:
            max_q_id = r[0]
        total_score += r[1]

    q_records = {}
    for r in records:
        qid = r[0]
        score = r[1]
        if qid not in q_records.keys():
            q_records[qid] = []

        q_records[qid].append(score)

    record_num = len(records)

    if next_question_id in q_records.keys():
        specific_avg_score = sum(q_records[next_question_id]) / len(q_records[next_question_id])
        question_diff = random.uniform(0, 4)
    else:
        specific_avg_score = 0
        question_diff = 2
    if len(records)>0:
        total_avg_score = total_score / len(records)
    else:
        total_avg_score = 0

    out = record_num / 100 + total_avg_score/2 + specific_avg_score * 2 - question_diff

    next_question_prediction = sigmoid(out)

    return next_question_prediction


def get_chart_input(records,data_range):

    num_elements = int(len(records) * (data_range / 100))
    records = records[:num_elements]
    skill_records = {}

    for r in records:
        qid = r[0]
        if r not in skill_records.keys():
            skill_records[qid] = []
    print(records)


    for i in range(len(records)):
        for qid in skill_records.keys():
            skill_records[qid].append(infer(records[:i], qid))
    skill_name_records = {}

    for s in skill_records.keys():
        skill_name_records['q' + s.__str__()] = skill_records[s]
    return skill_name_records



if __name__ == '__main__':
    dummy_records = [
        (1, 0), (1, 0), (1, 1),
        (2, 0), (2, 1), (2, 1),
        (3, 1), (3, 1), (3, 1)]
    print(infer(dummy_records, 1))
    print(infer(dummy_records, 2))
    print(infer(dummy_records, 3))

    print(get_chart_input(records=dummy_records,data_range=50))