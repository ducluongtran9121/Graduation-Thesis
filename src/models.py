import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#################################
# Models for federated learning #
#################################
# McMahan et al., 2016; 199,210 parameters
class TwoNN(nn.Module):
    def __init__(self, name, in_features, num_hiddens, num_classes):
        super(TwoNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.fc1 = nn.Linear(in_features=in_features, out_features=num_hiddens, bias=True)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_hiddens, bias=True)
        self.fc3 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# McMahan et al., 2016; 1,663,370 parameters
class CNN3(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN3, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        
        self.maxpool1 = nn.MaxPool1d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# for CIFAR10
class CNN2(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN2, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (8 * 8), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
    
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x

class CNN(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN, self).__init__()
        self.name = name
        # 1st convolutional layer
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding='same')    # 116*1 --> 116*64
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_channels)
        # self.dropout1 = nn.Dropout(0.5)

        # 2nd convolutional layer 
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels*2, kernel_size=3)                  # 116*64 --> 112*128
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_channels*2)
        # self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.dropout2 = nn.Dropout(0.5)
        
        # 3rd convolutional layer
        # self.conv3 = nn.Conv1d(in_channels=hidden_channels*2, out_channels=hidden_channels*4, kernel_size=3)                 # 112*128 --> 107 * 256
        # self.relu3 = nn.ReLU()
        # self.batchnorm3 = nn.BatchNorm1d(num_features=hidden_channels*4)
        # self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.dropout3 = nn.Dropout(0.5)

        # Fully Connected layer
        self.fc1 = nn.Linear(in_features=113*hidden_channels*2, out_features=num_hiddens)
        # self.fc3 = nn.Linear(in_features=num_hiddens, out_features=num_hiddens//4)
        self.fc4 = nn.Linear(in_features=num_hiddens, out_features=num_classes)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        # x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        # x = self.maxpool2(x)
        # x = self.dropout2(x)

        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.batchnorm3(x)
        # x = self.maxpool3(x)
        # x = self.dropout3(x)

        x = torch.flatten(x,1)
        x = self.fc1(x)
        # x = self.fc3(x)
        x = self.fc4(x)
        # out = self.sigmoid(x)

        return x

class LinearModel(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes)-> None:
        super(LinearModel, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(in_features=115, out_features=num_hiddens*2, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens*2, out_features=num_hiddens, bias=False)
        self.fc3 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(LSTM, self).__init__()
        self.name = name
        self.input_size = 115
        self.hidden_size = num_hiddens
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=115, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x.reshape(x.shape[0], 1, x.shape[1]), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class AE(nn.Module):
    def __init__(self) -> None:
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(14464, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True), 
            nn.Linear(128, 64), 
            nn.ReLU(True), 
            nn.Linear(64, 16)
            )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True), 
            nn.Linear(512, 14464), 
            nn.Tanh()
            )
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
