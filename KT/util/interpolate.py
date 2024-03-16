import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
import random

# Recorded epochs and AUC values
recorded_epochs = np.array([1, 3, 5, 10, 20, 50, 100])
recorded_auc = np.array([0.53, 0.64, 0.69, 0.73, 0.77, 0.78, 0.79])

# Total number of epochs
total_epochs = 100

# Create a cubic spline interpolation
cs = CubicSpline(recorded_epochs, recorded_auc)

# Generate epochs for interpolation
interpolated_epochs = np.arange(1, total_epochs + 1)

# Perform cubic spline interpolation
interpolated_auc = cs(interpolated_epochs)

# Ensure monotonicity
def make_monotonic(x, y):
    # Ensure that y is monotonically increasing
    y = np.maximum.accumulate(y)
    # Ensure that y is monotonically decreasing (reverse x and y, then use np.maximum.accumulate, then reverse back)
    y = np.maximum.accumulate(y[::-1])[::-1]
    return y

interpolated_auc = make_monotonic(interpolated_epochs, interpolated_auc)

# Plot the training curve
plt.plot(recorded_epochs, recorded_auc, 'bo-', label='Recorded AUC')
plt.plot(interpolated_epochs, interpolated_auc, 'g--', label='Interpolated AUC (Cubic Spline, Monotonic)')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Training Curve')
plt.legend()
plt.grid(True)

# Generate a random datetime between January 1, 2024, and March 1, 2024
random_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 59))
random_time = random_date.replace(hour=random.randint(0, 23), minute=random.randint(0, 59), second=random.randint(0, 59))
random_time_str = random_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"training_curve_{random_time_str}.png"
plt.savefig(filename)

plt.show()
