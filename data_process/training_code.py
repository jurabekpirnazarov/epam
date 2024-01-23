import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load the training data
features = pd.read_csv('training_features.csv')
target = pd.read_csv('training_target.csv')['target']

# Define the neural network model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(4, 3)  # 4 input features, 3 output classes

    def forward(self, x):
        x = self.fc(x)
        return x

# Instantiate the model
model = SimpleClassifier()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    inputs = torch.tensor(features.values, dtype=torch.float32)
    labels = torch.tensor(target.values, dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
