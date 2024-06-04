import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (1D array)
data = np.random.rand(20)

# Create a 2D array where each value in the original array corresponds to a row
heatmap_data = np.expand_dims(data, axis=0)

# Plot the heatmap
plt.figure(figsize=(10, 5))  # Adjust figure size as needed
plt.imshow(heatmap_data, cmap='Blues', aspect='auto')
plt.colorbar(label='Value', orientation='vertical')
plt.title('Blue-Color Heatmap for 1D Array')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()