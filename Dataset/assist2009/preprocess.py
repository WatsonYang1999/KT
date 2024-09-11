import pandas as pd
import numpy as np
from KT.util.decorators import suit_directory_decorator


@suit_directory_decorator()
def build_user_sequences():

    try:
        df = pd.read_csv('Dataset/assist2009/skill_builder_data.csv')[['order_id','user_id', 'problem_id','skill_id', 'skill_name','correct']]

        df = df.sort_values('order_id',ascending=True).dropna().drop_duplicates(subset='order_id')

        cols = ['order_id', 'assignment_id', 'user_id', 'assistment_id', 'problem_id',
                'original', 'correct', 'attempt_count', 'ms_first_response',
                'tutor_mode', 'answer_type', 'sequence_id', 'student_class_id',
                'position', 'type', 'base_sequence_id', 'skill_id', 'skill_name',
                'teacher_id', 'school_id', 'hint_count', 'hint_total', 'overlap_time',
                'template_id', 'answer_id', 'answer_text', 'first_action',
                'bottom_hint', 'opportunity', 'opportunity_original']
        for col in ['problem_id', 'skill_id']:
            df[col] = df[col].astype(int)
    except Exception as e:
        print(e)
        exit(-1)


    df_user_groups = df.groupby('user_id')

    seqs = {}

    for uid, df_user in df_user_groups:

        assert uid == df_user['user_id'].iloc[0]
        # are the interactions following the time order

        q_seq = df_user['problem_id'].tolist()
        r_seq = df_user['correct'].tolist()

        seqs[uid] = {"question": q_seq, "result": r_seq}

    return seqs

@suit_directory_decorator()
def load_qs_relation():

    df = pd.read_csv('Dataset/assist2009/skill_builder_data.csv')

    df = df[['problem_id', 'skill_id', 'skill_name']].drop_duplicates().dropna()

    qs_mapping = {}
    sq_mapping = {}
    for idx, row in df.iterrows():


        qid = row['problem_id']

        sid_set = set()

        sid_set.add(int(row['skill_id']))

        for sid in sid_set:
            if sid not in sq_mapping.keys():
                sq_mapping[sid] = set()
            sq_mapping[sid].add(qid)

            if qid not in qs_mapping.keys():
                qs_mapping[qid] = set()
            qs_mapping[qid].add(sid)

    # print(qs_mapping)
    # print(sq_mapping)

    # test_case1 = {25868:[3, 49, 62, 64, 70, 154]}
    # for tc in test_case1:
    #     q = tc
    #     s_list = test_case1[q]
    #     for s in s_list:
    #         assert q in sq_mapping[s]
    #         assert s in qs_mapping[q]
    print('load qs relation done and pass')
    return qs_mapping, sq_mapping

if __name__ == '__main__':
    df = pd.read_csv('skill_builder_sorted_by_order_id.csv')
    df_origin = pd.read_csv('skill_builder_data.csv')[['order_id','user_id', 'problem_id','skill_id', 'skill_name','correct']]
    df_origin = df_origin.sort_values(by='order_id')
    df_origin = df_origin.dropna()
    print(f'unique user count: {len(df.user_id.unique()) }')
    print(f'unique user count: {len(df_origin.user_id.unique()) }')
    exit(-1)
    print(build_user_sequences())
    print(load_qs_relation())