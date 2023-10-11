import matplotlib.pyplot as plt

# 数据
x = [51, 52, 53, 54, 55]  # x 轴数据
y1 = [0.12, 0.44, 0.67, 0.76, 0.88]  # 第一条折线的 y 轴数据
y2 = [0.31, 0.43, 0.44, 0.54, 0.61]  # 第二条折线的 y 轴数据

# 创建一个图形
plt.figure()

# 绘制第一条折线
plt.plot(x, y1, label='q_x', marker='o')

# 绘制第二条折线
plt.plot(x, y2, label='q_y', marker='s')

# 添加图例
plt.legend()

# 设置图形标题和坐标轴标签
plt.title('mastery relation')
plt.xlabel('time')
plt.ylabel('score')

# 显示图形
plt.show()
