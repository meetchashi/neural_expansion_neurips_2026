import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1, hidden_channels1=32, hidden_channels2=64, 
                 input_height=28, input_width=28):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels1)
        self.conv2 = nn.Conv2d(hidden_channels1, hidden_channels2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the feature size based on the provided input dimensions
        # After 2 pooling layers with stride 2, the spatial dimensions are reduced by factor of 4
        feature_height = input_height // 4
        feature_width = input_width // 4
        feature_size = hidden_channels2 * feature_height * feature_width
        
        # Initialize fc layers at creation time
        self.fc1 = nn.Linear(feature_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Final output dimension
        self.num_classes = num_classes
        
    def forward(self, x):
        # First convolutional layer
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second convolutional layer
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def extract_features(self, x):
        """
        Extracts features from each layer of the network.
        Returns a list of tensors containing:
        - Input image
        - First conv+pool output
        - Second conv+pool output
        - Flattened features
        - First FC layer output (penultimate layer)
        - Final output
        """
        features = []
        
        # Store input
        features.append(x)
        
        # First conv layer + pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        features.append(x)
        
        # Second conv layer + pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        features.append(x)
        
        # Flatten
        x_flat = x.view(x.size(0), -1)
        features.append(x_flat)
        
        # FC1 (penultimate layer)
        x = F.relu(self.fc1(x_flat))
        features.append(x)
        
        # FC2 (final output)
        x = self.fc2(x)
        features.append(x)
        
        return features
