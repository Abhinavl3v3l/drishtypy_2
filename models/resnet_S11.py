import torch.nn as nn
import torch.nn.functional as F


class ResNet_11(nn.Module):
    def __init__(self):
        super(ResNet_11, self).__init__()
        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.prepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Define Convolutions to calculate Layer1
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.R1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # Define Convolutions for Layer2
        # Conv 3x3 [256k] -MaxPooling2D-BN-ReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Define Convolutions to calculate Layer3
        # Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.x2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        self.R2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # MaxPooling with Kernel Size 4
        self.pool = nn.MaxPool2d(4, 4)
        # FC Layer
        self.fc = nn.Linear(in_features=512, out_features=10, bias=False)

    def forward(self, x):
        # prepLayer
        prepLayer = self.prepLayer(x)
        # Add(X, R1)
        x1 = self.x1(prepLayer)
        R1 = self.R1(x1)
        layer1 = x1 + R1

        layer2 = self.layer2(layer1)
        x2 = self.x2(layer2)
        R2 = self.R2(x2)

        # Add(X2, R2)
        layer3 = R2 + x2

        maxpool = self.pool(layer3)
        x = maxpool.view(maxpool.size(0), -1)

        fc = self.fc(x)
        return F.log_softmax(fc.view(-1, 10), dim=-1)