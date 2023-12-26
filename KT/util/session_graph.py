import numpy as np
import matplotlib.pyplot as plt

def generate_session_graph_matrix(browsing_history):
    n = max(browsing_history)  # Assuming item IDs are integers
    session_matrix = np.zeros((n+1, n+1), dtype=int)

    for i in range(len(browsing_history) - 1):
        current_item = browsing_history[i]
        next_item = browsing_history[i + 1]
        session_matrix[current_item, next_item] += 1

    return session_matrix

# Example browsing history
browsing_history = [1, 2, 3, 1, 4, 2, 5, 1]

# Generate session graph matrix
session_matrix = generate_session_graph_matrix(browsing_history)

# Display the session matrix
print("Session Graph Matrix:")
print(session_matrix)

# Plot the session graph (optional)
plt.imshow(session_matrix, cmap='Blues', interpolation='none')
plt.title('Session Graph Matrix')
plt.xlabel('Next Item ID')
plt.ylabel('Current Item ID')
plt.colorbar()
plt.show()