import torch

# 创建一个示例矩阵
matrix = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

# 创建一个与矩阵相同形状的掩码矩阵
mask = (matrix % 2 == 0)

print(mask)

# 将矩阵中对应掩码为True的元素修改为指定值
new_value = 0
matrix[mask] = new_value

# 打印修改后的矩阵
print(matrix)
