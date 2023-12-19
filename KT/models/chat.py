import torch
import torch.nn as nn

class DKT(nn.Module):
    def __init__(self, num_skills, hidden_dim):
        super(DKT, self).__init__()

        self.embedding = nn.Embedding(num_skills * 2, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, h_n = self.gru(embedded)
        output = self.fc(h_n[-1])  # Take the last hidden state
        return torch.sigmoid(output)

# Example usage:
num_skills = 10  # Adjust based on your dataset
hidden_dim = 64  # Adjust based on your model complexity

# Instantiate the DKT model
model = DKT(num_skills, hidden_dim)

# Example input sequence (batch_size=1, sequence_length=5)
input_sequence = torch.tensor([[1, 3, 5, 2, 4]])

# Forward pass
output = model(input_sequence)

# Print the output
print(output)
