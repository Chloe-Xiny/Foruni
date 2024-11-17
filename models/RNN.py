import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(SimpleRNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully Connected Layer for Classification
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.act = F.relu
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through RNN
        out, _ = self.rnn(x, h0)
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Apply activation function
        out = self.act(out)
        
        # Pass through fully connected layer for classification
        out = self.fc(out)
        
        return out

# Example usage
# Define model parameters
input_size = 448  # Dimension of input features (based on merged tensor size)
hidden_size = 64  # Number of hidden units in the RNN
output_size = 3   # Number of classes for classification
num_layers = 2    # Number of RNN layers

dropout = 0.3

# Instantiate the model
model = SimpleRNNClassifier(input_size, hidden_size, output_size, num_layers, dropout)

# Print the model architecture
print(model)
