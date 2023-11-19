from __future__ import annotations
import pandas as pd
from collections import defaultdict
import numpy as np
from typing import List
from dataclasses import dataclass, field
from pyvis.network import Network


def simple_observe(df: pd.DataFrame):
    print(df.columns)
    print(df.head(10))
    print(df.describe())


def build_user_sequences(df: pd.DataFrame):
    assert len(df['AnswerValue'].unique()) == 4
    assert len(df['CorrectAnswer'].unique()) == 4

    df_user_groups = df.groupby('UserId')

    seqs = {}

    for uid, df_user in df_user_groups:
        assert uid == df_user['UserId'].iloc[0]
        seq_len = len(df_user)
        # are the interactions following the time order

        q_seq = df_user['QuestionId'].tolist()
        r_seq = df_user['IsCorrect'].tolist()
        ca_seq = df_user['CorrectAnswer'].tolist()
        sa_seq = df_user['AnswerValue'].tolist()

        seqs[uid] = {"question": q_seq, "result": r_seq, "ca": ca_seq, "sa": sa_seq}

    return seqs


def load_qs_relations():
    df = pd.read_csv(r'C:\Users\12574\Documents\GitHub\Ednet-KT\data\metadata\question_metadata_task_1_2.csv')
    print(df)
    qs_mapping = {}
    sq_mapping = {}
    for idx, row in df.iterrows():
        qid = row['QuestionId']
        import ast
        sid_set = ast.literal_eval(row['SubjectId'])

        for sid in sid_set:
            if sid not in sq_mapping.keys():
                sq_mapping[sid] = set()
            sq_mapping[sid].add(qid)

            if qid not in qs_mapping.keys():
                qs_mapping[qid] = set()
            qs_mapping[qid].add(sid)

    # print(qs_mapping)
    # print(sq_mapping)
    return qs_mapping, sq_mapping


def load_knowledge_structure():
    df = pd.read_csv(r'C:\Users\12574\Documents\GitHub\Ednet-KT\data\metadata\subject_metadata.csv')
    print(df.columns)
    # df['SubjectId'] = df['SubjectId'].astype(int)
    # df['ParentId'] = df['ParentId'].astype(int)
    # create graph
    from dataclasses import dataclass
    @dataclass
    class SkillNode:
        sid: int
        sname: str
        level: int
        parent_id: int
        children: List = field(default_factory=lambda: [])

        def add_child(self, child_node):
            self.children.append(child_node)

    @dataclass
    class SkillTopoGraph:
        from pyvis.network import Network
        nodes: dict = defaultdict
        root: SkillNode = None
        vis_graph: Network = Network()

        def build(self, df: pd.DataFrame):
            self.nodes = {}
            for idx, row in df.iterrows():
                sid = int(row['SubjectId'])
                s_name = row['Name']
                if pd.isna(row['ParentId']):
                    s_parent_id = None
                else:
                    s_parent_id = int(row['ParentId'])
                level = row['Level']

                self.nodes[sid] = SkillNode(sid, s_name, level, s_parent_id)

            for idx, row in df.iterrows():
                sid = int(row['SubjectId'])

                if pd.isna(row['ParentId']):
                    self.root = self.nodes[sid]
                else:
                    s_parent_id = int(row['ParentId'])
                    parent_node = self.nodes[s_parent_id]
                    parent_node.add_child(self.nodes[sid])

        def level_order_traversal(self):

            print(f'total nodes {len(self.nodes)}')

            leaves = []
            for node in self.nodes.values():

                if len(node.children) == 0:
                    leaves.append(node)
            print(f'leaves count {len(leaves)}')

            if not self.root:
                return []
            result = []
            queue = [self.root]

            while queue:
                current_node = queue.pop(0)

                result.append(current_node.sid)
                queue.extend(current_node.children)

            return result

        def check_node_level_align(self):
            pass

        def visualize(self):
            def visualize_tree(node, graph):
                print(node, node.children)
                if node is not None:
                    graph.add_node(node.sid, label=node.sname)
                    for child in node.children:
                        assert node.sid in self.nodes.keys()
                        assert child.sid in self.nodes.keys()
                        try:
                            graph.add_edge(node.sid, child.sid)

                        except Exception as e:
                            graph.add_node(child.sid, label=child.sname)

                        print(node.sid, child.sid)
                        visualize_tree(child, graph)

            def wtf(graph):
                skill_freq = frequency_count()
                max_frequency = max(skill_freq.values())
                for idx, row in df.iterrows():
                    sid = int(row['SubjectId'])
                    s_name = row['Name']

                    color = f'rgba(255, 0, 0, {skill_freq[sid] / max_frequency/2 + 0.5})'

                    graph.add_node(sid, label=s_name, color=color)

                for idx, row in df.iterrows():
                    sid = int(row['SubjectId'])
                    if not pd.isna(row['ParentId']):
                        s_parent_id = int(row['ParentId'])
                        graph.add_edge(s_parent_id, sid)

            graph = Network(directed=True)
            wtf(graph)
            # graph.save_graph("tree_visualization.html")
            graph.show("tree_visualization.html")

    skillgraph = SkillTopoGraph()
    skillgraph.build(df)
    # print(skillgraph.level_order_traversal())
    skillgraph.visualize()


def frequency_count():
    df = pd.read_csv(r'C:\Users\12574\Documents\GitHub\Ednet-KT\data\train_data\train_task_1_2.csv')
    seqs = build_user_sequences(df)
    qs_mapping, sq_mapping = load_qs_relations()

    skill_freq = {}
    for uid, seq in seqs.items():

        for q in seq['question']:
            for s in qs_mapping[q]:
                if s in skill_freq:
                    skill_freq[s] += 1
                else:
                    skill_freq[s] = 1
    print(skill_freq.keys())
    print(skill_freq)
    return skill_freq


if __name__ == '__main__':
    # simple_observe(df_train)
    # build_user_sequences(df_train)
    # load_qs_relations()
    load_knowledge_structure()
    # frequency_count()
