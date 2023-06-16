
import pandas as pd

pre = ['2018_w','2018_s','2019_w','2019_s']


#merge dict
columns = ['id', 'index', 'io', 'format', 'calculate', 'if', 'for', 'list', 'map',
       'string', 'stack', 'queue', 'enumeration', 'dp', 'sort', 'search',
       'bit']
KCs =  ['io', 'format', 'calculate', 'if', 'for', 'list', 'map',
       'string', 'stack', 'queue', 'enumeration', 'dp', 'sort', 'search',
       'bit']
df = None
for p in pre:
    df_i = pd.read_excel(p+'_exercise.xlsx')


    for c in columns:
        if c not in df_i.columns:
            new_col = [0 for i in range(df_i.shape[0])]
            df_i[c] = new_col
    for c in df_i.columns:
        if c not in columns:
            print(c)
            print(type(c))
            df_i  = df_i.drop(columns=[c])
    if df is None:
        df = df_i
    else:
        df = pd.concat([df,df_i])
    print(df_i.shape)
print(df.shape)
df = df.reset_index()
print(df)
useful =['id','io', 'format', 'calculate', 'if', 'for', 'list', 'map',
       'string', 'stack', 'queue', 'enumeration', 'dp', 'sort', 'search',
       'bit']
df = df[useful]
print(df)
df.to_csv('q_matrix.csv')