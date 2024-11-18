import torch
import torch.nn as nn
import torch.optim as optim

class SimpleLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(SimpleLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        out = out[:, -1, :]

        # Pass through the fully connected layer
        out = self.fc(out)
        return out

# Example usage
def main():
    # Hyperparameters
    input_size = 448  # Number of features per time step
    hidden_size = 64  # Number of LSTM units
    num_layers = 2  # Number of LSTM layers
    num_classes = 3  # Number of output classes
    sequence_length = 800  # Number of time steps (frame length)
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001

    # Create the model
    model = SimpleLSTMClassifier(input_size, hidden_size, num_layers, num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy data for demonstration purposes
    inputs = torch.randn(batch_size, sequence_length, input_size)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()
