from torch import nn


class LeNet5(nn.Module):
    def __init__(self,
                 num_classes,
                 gray_scale=True):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = 1 if gray_scale else 3

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      kernel_size=5,
                      out_channels=6),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      kernel_size=5,
                      out_channels=16),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      kernel_size=5,
                      out_channels=120)
        )

        self.fc_layer_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=120,
                      out_features=84),
            nn.Tanh()
        )

        self.fc_layer_2 = nn.Sequential(
            nn.Linear(in_features=84,
                      out_features=self.num_classes),
        )

    def forward(self, x):
        # print(f'Initial Shape: {x.shape}')
        x = self.conv_layer_1(x)
        # print(f"After passing through Conv 1: {x.shape}")
        x = self.conv_layer_2(x)
        # print(f"After passing through Conv 2: {x.shape}")
        x = self.conv_layer_3(x)
        # print(f"After passing through Conv 3: {x.shape}")
        x = self.fc_layer_1(x)
        # print(f"After passing through FC 1: {x.shape}")
        x = self.fc_layer_2(x)
        # print(f"After passing through FC 2: {x.shape}")
        return x
