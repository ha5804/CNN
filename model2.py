import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, num_classes = 4):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 3 * 3 , num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


