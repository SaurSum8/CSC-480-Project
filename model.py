from torch import nn

class CNN(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, p=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(0.15)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(0.15)

        self.fc1 = nn.Linear(256 * 2 * 16, 1024)
        self.fc_drop1 = nn.Dropout(0.30)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_drop2 = nn.Dropout(0.30)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.drop1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(self.relu(self.bn4(self.conv4(x)))))

        x = x.view(x.size(0), -1)
        x = self.fc_drop1(self.relu(self.fc1(x)))
        x = self.fc_drop2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
