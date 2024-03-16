import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to simulate mastery scores over time with random walk
def simulate_mastery_over_time(num_skills, num_problems, num_iterations, initial_mastery=0.5, noise_level=0.1, min_threshold=0.25):
    mastery_data = np.full((num_skills, num_problems), initial_mastery)
    for _ in range(num_iterations):
        mastery_data += np.random.normal(0, noise_level, (num_skills, num_problems))
        mastery_data = np.clip(mastery_data, min_threshold, 1)  # Clip values to be within [min_threshold, 1]
    return mastery_data

# Parameters
num_skills = 5
num_problems = 20
num_iterations = 10  # Increase this number for more time steps
initial_mastery = 0.5
noise_level = 0.1
min_threshold = 0.25

# Simulate mastery scores over time
mastery_data = simulate_mastery_over_time(num_skills, num_problems, num_iterations, initial_mastery, noise_level, min_threshold)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(mastery_data, cmap='YlGnBu', annot=True, fmt=".2f", cbar=True, linewidths=0.5)
plt.title('Student Mastery of Five Skills Over Time')
plt.xlabel('Practice Problems')
plt.ylabel('Skills')
plt.xticks(np.arange(0.5, num_problems + 0.5, 1), range(1, num_problems + 1))
plt.yticks(np.arange(0.5, num_skills + 0.5, 1), ['Skill {}'.format(i) for i in range(1, num_skills + 1)], rotation=0)
plt.tight_layout()
plt.show()
