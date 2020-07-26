import torch
import torch.nn as nn



class SPCNNet(nn.Module):
    def __init__(self, upscale_factor):
        super(SPCNNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.sub_pixel = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.sub_pixel(self.conv3(x)))
        return x



