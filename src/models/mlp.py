import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a plain neural network
class PlainNN(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim1=256, hidden_dim2=256, output_dim=10):
        super(PlainNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # Flatten input (28x28 image size)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0)  # Drop 10% of neurons

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image (28x28 to 1D)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)  # Output layer
        
        return x

    def extract_features(self, x):
        """
        Extracts the features from the penultimate layer (fc2).
        """
        features = []
        x = x.view(x.size(0), -1)  # Flatten the image (28x28 to 1D)
        features.append(x)
        x = self.relu(self.fc1(x))
        features.append(x)
        x = self.relu(self.fc2(x))  # Features before the final layer
        features.append(x)
        x = self.fc3(x)  # Output layer
        features.append(x)
        return features  # Return the features from fc2
