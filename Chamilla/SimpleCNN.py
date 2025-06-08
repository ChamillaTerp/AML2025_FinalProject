import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Defining a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_outputs = 8):                                        # Initialize the CNN with a specified number of output classes (default is 8)

        super(SimpleCNN, self).__init__()                                       # Initialize the parent class 

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)             # First convolutional layer with 3 input channels (RGB), 16 output channels, and a kernel size of 3
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)            # Second convolutional layer with 16 input channels and 32 output channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)            # Third convolutional layer with 32 input channels and 64 output channels

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)                   # Max pooling layer to reduce spatial dimensions 
        self.dropout = nn.Dropout(0.5)                                          # Dropout layer to prevent overfitting - randomly sets 50% of the input units to 0 during training

        self.fc1 = nn.Linear(64 * 28 * 28, 256)                                 # Fully connected layer with input size based on the output of the last convolutional layer and 256 output units
        self.fc2 = nn.Linear(256, num_outputs)                                  # Second fully connected layer with 256 input units and output size equal to the number of labels

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))                                    # Apply first convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))                                    # Apply second convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv3(x)))                                    # Apply third convolution, ReLU activation, and max pooling

        x = x.view(x.size(0), -1)                                                # Flatten the tensor for the fully connected layers
        x = F.relu(self.fc1(x))                                                  # Apply first fully connected layer with ReLU activation
        x = self.dropout(x)                                                      # Apply dropout
        x = self.fc2(x)                                                          # Apply second fully connected layer

        return x                                                                 # Return the output logits