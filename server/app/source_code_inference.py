import openai
import requests
import json
# Set your OpenAI API key
def get_prompt(code_fragment):
    prompt =  f"""
    Classify the following code fragment into the appropriate tags from the given categories.
    The output should only includes the tags in a list,the tag should be wrapped in <> .
    Categories and tags:

    Programming Basics: Array, Branching Statements, Constants and Variables, Basic Data Types, Constant, Variable Initialization, Variable Assignment, Variable Types, Custom Types, Dynamic Memory Management, Functions, Custom Functions, Library Functions, IO Functions, String Functions, Recursive Functions, Loop Statements, Operators and Expressions, Arithmetic Operators, Bitwise Operators, Expression Evaluation, Implicit Type Conversion, Addressing Operators, Logical Operators, Relational Operators, Assignment Operators, Conditional Operators, Pointers, Preprocessing, Predefined Symbols, Conditional Compilation, File Inclusion, Macros
    Data Structures: Linear Structures, List, Matrix, Triangular Matrix, Queue, Stack, String, Tree Structures, Basic Tree Terminology, Forest, Tree, Binary Tree, Multiway Tree, Tree Storage Structures, Graph, Basic Graph Terminology, Directed Graph, Undirected Graph, Activity Network, Activity Edge Network, Graph Storage Structures, Graph Traversal, Minimum Spanning Tree
    Algorithms: Shortest Path, Search, Basic Search Terminology, Hash Search, Linear Search, Tree Search, Sorting, Bubble Sorting 
    External Sorting, Internal Sorting, Insertion Sorting, Merge Sorting, Radix Sorting, Selection Sorting,Quick Sorting,
    DFS, BFS,Greedy,Dynamic Programming,Backtracking.
    Code fragment:
    {code_fragment}

    Tags:
    """
    return prompt

def source_code_inference(prompt):
    url = "https://www.jiujiuai.life/v1/chat/completions"


    headers = {
        "Content-Type": "application/json",

        "Authorization": "sk-omSzZiaMUYbDD91f1b3a25Fe29394217Bd57C53dD84c8cBc"

    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    '''
    {'id': 'chatcmpl-ObwSiOMOSU2qod3Cjm89MiEalogAZ', 'model': 'gpt-3.5-turbo', 'object': 'chat.completion', 'created': 1717478292, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': "To solve this programming question, you'll need:\n\nBasic programming skills:\n- Understanding of loops (for loop)\n- Basic string manipulation (formatting output)\n\nData structures and algorithms:\n- None required, as this task doesn't involve complex data structures or algorithms.\n\nBe specific:\n- Print the multiplication tables from 1 to 9 in the specified format."}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 236, 'completion_tokens': 71, 'total_tokens': 307}}
    '''
    response = requests.post(url, headers=headers, json=data)

    output = json.loads(response.text)
    return output['choices'][0]['message']['content']

def extract_content_between_brackets(input_string):
    import re
    # Use regex to find all content between < and >
    matches = re.findall(r'<(.*?)>', input_string)
    return matches

def get_substring_after_delimiters(input_string):
    import re
    # Use regex to find the substring after '-' or ':'
    match = re.search(r'[-:](.*)', input_string)
    if match:
        return match.group(1).strip()
    return input_string

def get_tags(source_code):
    try:
        # Get the classification result
        code_prompt = get_prompt(source_code)
        tags = source_code_inference(code_prompt)
        tags = extract_content_between_brackets(tags)
        print(tags)
        for i in range(len(tags)):
            tags[i] = get_substring_after_delimiters(tags[i])
        result = []
        for t in tags:
            if t and t[0].isupper():
                result.append(t)

        return result
    except Exception as e:
        print('Error Occurred , Try Again')

if __name__=='__main__':
    code_snippet = """
        def add(a, b):
            return a + b
        """
    print(get_tags(code_snippet))