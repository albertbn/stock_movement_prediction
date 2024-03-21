import torch
import torch.nn.functional as F
from torch import nn, cuda
from DataScale import DataScale

NUM_NUMERICAL_COLS = 6
TWEET_EMBEDDING_DIM = 1536


class ModelCnn(DataScale):

    def __init__(self):
        super().__init__()
        n_symbols = len(self.symbol_to_code)
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        print(f'device is: {self.device}')
        self.model = StockCNN(NUM_NUMERICAL_COLS+TWEET_EMBEDDING_DIM).to(self.device)


class StockCNN(nn.Module):
    def __init__(self, num_features):
        super(StockCNN, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Define fully connected layers
        # The size 128 * 2 needs to be adjusted based on the output shape of the last conv layer
        self.fc1 = nn.Linear(128 * 2, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, src_data: torch.Tensor, src_symbol: torch.Tensor, src_tweet: torch.Tensor):
        # Input shape x: [batch, channels, width] where channels=num_features and width=sequence length

        x = torch.cat((src_data, src_tweet), dim=-1)

        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x.transpose(1,2))))  # -> 32, 64, 5
        x = self.pool(F.relu(self.conv2(x)))  # -> 32, 128, 2

        # Flatten the output for the dense layer
        x = torch.flatten(x, 1)  # -> 32, 256

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))  # Output layer for regression

        return x


# Assuming the combined feature size is the original features plus the embedding size
# num_features = 6 + embedding_size  # 6 for Open, High, Low, Close, Adj Close, Volume, and embedding size
# model = StockCNN(num_features=num_features)
# print(model)
