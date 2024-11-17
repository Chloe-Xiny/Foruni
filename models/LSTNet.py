import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTNetClassifier(nn.Module):
    def __init__(self, input_dim, hidden_cnn, kernel_size, hidden_rnn, output_dim, window_size, dropout=0.2):
        super(LSTNetClassifier, self).__init__()
        
        # Convolutional Layer to capture local dependencies
        self.conv1 = nn.Conv2d(1, hidden_cnn, kernel_size=(kernel_size, input_dim))
        
        # Gated Recurrent Unit (GRU) to capture temporal patterns
        self.gru = nn.GRU(hidden_cnn * (window_size - kernel_size + 1), hidden_rnn)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_rnn, output_dim)
        
        # Dropout layer to avoid overfitting
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input x shape: (batch_size, window_size, input_dim)
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # Reshape to (batch_size, 1, window_size, input_dim) for CNN
        
        # Convolutional layer
        c = F.relu(self.conv1(x))  # Output shape: (batch_size, hidden_cnn, new_window_size, 1)
        c = c.squeeze(3)  # Remove the last dimension, shape: (batch_size, hidden_cnn, new_window_size)
        
        # Reshape for GRU
        c = c.permute(2, 0, 1)  # Shape: (new_window_size, batch_size, hidden_cnn)
        
        # GRU layer
        r_out, _ = self.gru(c)  # Output shape: (new_window_size, batch_size, hidden_rnn)
        r_out = r_out[-1, :, :]  # Get the last output for classification, shape: (batch_size, hidden_rnn)
        
        # Dropout
        r_out = self.dropout(r_out)
        
        # Fully connected layer
        out = self.fc(r_out)  # Shape: (batch_size, output_dim)
        
        return out

# Example usage
if __name__ == "__main__":
    # Define model parameters
    input_dim = 800  # Number of features
    hidden_cnn = 16  # Number of output channels for CNN
    kernel_size = 3  # Kernel size for CNN
    hidden_rnn = 32  # Number of hidden units for GRU
    output_dim = 2  # Number of classes for classification
    window_size = 12  # Length of the input sequence (time window)
    dropout = 0.3  # Dropout rate

    # Instantiate the model
    model = LSTNetClassifier(input_dim, hidden_cnn, kernel_size, hidden_rnn, output_dim, window_size, dropout)
    
    # Create a sample input: (batch_size, window_size, input_dim)
    sample_input = torch.randn(8, window_size, input_dim)  # Batch size of 8
    
    # Forward pass
    output = model(sample_input)
    print(output.shape)  # Expected output shape: (8, output_dim)
