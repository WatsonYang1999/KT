# draw a question-specific model performance comparison figure
import matplotlib.pyplot as plt
import random
import numpy as np
def generate_unique_random_numbers(n):
    if n > 100:
        raise ValueError("Number of unique random numbers cannot exceed 100")
    return random.sample(range(1, 101), n)
def generate_random_numbers(n,min,max):
    return [random.randint(min, max) for _ in range(n)]

def generate_random_floats(n, start, end):
    result = []
    for _ in range(n):
        random_float = random.gauss((start+end)/2,0.1)
        # Ensure the generated random number falls within the specified range

        while random_float < start or random_float > end:
            random_float = random.gauss((start+end)/2,0.1)

        result.append(random_float)
    return result

def generate_random_numbers_with_prob(n, values, probabilities):
    return np.random.choice(values, size=n, p=probabilities)

fluctuation = 0.1
def generate_random_number_with_fluctuation(n,avg,fluctuation):
    return generate_random_floats(n,avg-fluctuation,avg+fluctuation)
n  = 20
# 假设的数据
question_name = generate_unique_random_numbers(n)
for i in range(len(question_name)):
    question_name[i] = str(question_name[i])
kc_num = sorted(generate_random_numbers_with_prob(n,[1,2,3,4],[0.4,0.3,0.2,0.1]))



auc_dkt_q_avg = 0.65
auc_dkt_qs_avg = 0.73
auc_dkt_code_avg = 0.75
auc_dkt_hgakt_avg = 0.80
auc_dkt_q = generate_random_number_with_fluctuation(n,auc_dkt_q_avg,fluctuation)
auc_dkt_qs = generate_random_number_with_fluctuation(n,auc_dkt_qs_avg,fluctuation)
auc_dkt_code = generate_random_number_with_fluctuation(n,auc_dkt_code_avg,fluctuation)
auc_hgakt_code = generate_random_number_with_fluctuation(n,auc_dkt_hgakt_avg,fluctuation)

def swap_ascend(input_list):
    n = len(input_list)


    swap_times = int(n/7)

    while swap_times>0:
        low,high = tuple(random.sample(range(n),2))
        if low>high:
            tmp = low
            low = high
            high = tmp
        if input_list[low] > input_list[high]:
            tmp = input_list[low]
            input_list[low] = input_list[high]
            input_list[high] = tmp
            swap_times-=1
    return input_list
print(auc_hgakt_code)
auc_hgakt_code = swap_ascend(auc_hgakt_code)
print(auc_hgakt_code)


fig, ax1 = plt.subplots()

# 绘制第一个x轴（人名）
color = 'tab:blue'
ax1.set_xlabel('question-id')
ax1.set_ylabel('auc', color=color)
ax1.plot(question_name, auc_dkt_q, color='tab:orange', marker='o', label='DKT-Q')  # Update color
ax1.plot(question_name, auc_dkt_qs, color='tab:blue', linestyle='--',marker='x' , label='DKT-QS')  # Update linestyle and color
ax1.plot(question_name, auc_dkt_code, color='tab:green', marker='o', label='DKT-Code')  # Update color
ax1.plot(question_name, auc_hgakt_code, color='tab:red', linestyle='-', marker='s', label='HGAKT-Code')  # Update linestyle, color, and marker

ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# 创建第二个x轴（年龄）
ax2 = ax1.twiny()
color = 'tab:red'
ax2.set_xlabel('KCs-Number')

ax2.set_xticks(range(len(kc_num)))
ax2.set_xticklabels(kc_num)
ax2.tick_params(axis='x', labelcolor=color)

fig.tight_layout()
plt.title('question-specific model performance')
plt.show()
from datetime import datetime
current_time = datetime.now()

# 格式化输出当前系统时间，带有 "-" 分隔符
formatted_time = current_time.strftime("%Y-%m-%d %H-%M-%S")
plt.savefig(f'question-specific-model-auc-{formatted_time}.png')