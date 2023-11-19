import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 真实标签和预测概率
true_labels = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
predicted_probabilities = np.array([0.2, 0.3, 0.7, 0.8, 0.4, 0.9, 0.1, 0.95, 0.85, 0.3])

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)

# 计算AUC
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
