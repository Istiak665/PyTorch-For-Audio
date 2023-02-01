import torch
from torch import nn
from torchsummary import summary


# Create class for CNNModel
class CNNSoundRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 conv blocks -> flatten -> linear layer -> sigmoid/softmax

        # First Convolution Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=2,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Second Convolution Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=2,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Third Convolution Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=2,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Fourth Convolution Block
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=2,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Flatten Layer
        self.flatten = nn.Flatten()
        # Linear Layer
        self.linear = nn.Linear(in_features=128 * 5 * 4, out_features=14) # For 22050
        # self.linear = nn.Linear(in_features=128 * 5 * 3, out_features=14) # For 16000
        # Sigmoid Layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.sigmoid(logits)
        return predictions


if __name__ == "__main__":

    # Device agnostic code for GPU and CPU
    device = "CUDA" if torch.cuda.is_available() else "CPU"

    csr = CNNSoundRecognizer()
    # For sample rate 22050
    summary(csr, (1, 64, 44))
    # For sample rate 16000
    # summary(csr, (1, 64, 32))

# Output Shape Calculation
# Output shape = ((Input Size - Kernel + Padding * 2)/Stride) + 1
# Param = (width of filter*height filter*number of filters in the previous layer+1)*number of filters current layer

# For example in first Conv1:
# Param = (3*3*1+1)*16
#       = 10*16
#       = 160