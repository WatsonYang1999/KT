import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['Assistment09', 'Assistment12', 'Junyi', 'EdNet']
models = ['HGAKT', 'AKT', 'DKT', 'GKT']
auc_values = np.random.uniform(0.7, 0.82, (len(datasets), len(models)))


# Bar width
bar_width = 0.15

# Set position of bar on X axis
r = np.arange(len(datasets))

# Plotting
fig, ax = plt.subplots()

for i, model in enumerate(models):
    ax.bar(r + i * bar_width, auc_values[:, i], bar_width, label=model)

# Adding labels
ax.set_xlabel('Dataset')
ax.set_ylabel('AUC')
ax.set_title('AUC scores by Dataset and Model')
ax.set_xticks(r + bar_width * (len(models) - 1) / 2)
ax.set_xticklabels(datasets)
ax.legend()

plt.tight_layout()
plt.show()
