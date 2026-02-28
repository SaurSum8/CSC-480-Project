from torch import nn


class CNN(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, p=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 1))
        self.drop3 = nn.Dropout2d(p)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1))
        self.drop4 = nn.Dropout2d(p)

        self.lstm = nn.LSTM(
            input_size=256 * 2,
            hidden_size=256,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )

        self.fc1 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.drop3(self.pool3(self.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(self.relu(self.bn4(self.conv4(x)))))

        n, c, h ,w = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(n, w, c*h)
        h_n, c_n = self.lstm(x) # returns hidden states, cell states
        x = self.fc1(h_n)

        return x
