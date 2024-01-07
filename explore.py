import numpy as np


def generate_nxn_array(input_array):
    n = 5
    result_array = -1 * np.ones_like(input_array)

    result_array = np.tile(result_array,(n,1))
    print(result_array)
    for i in range(n):
        print(i)
        result_array[i, :i + 1] = input_array[:i + 1]

    return result_array


# 示例输入一维数组
input_array = np.array([1, 2, 3, 4, 5,0,0,0])

# 生成n*n数组
result = generate_nxn_array(input_array)

print(result)