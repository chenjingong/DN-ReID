import torch.nn as nn
import torch

class GradientComputation4(nn.Module):
    def __init__(self):
        super(GradientComputation4, self).__init__()
        self.conv_layers = {}
        self.relu = nn.ReLU()

    def get_masked_conv(self, channels):
        if channels not in self.conv_layers:
            mask_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, padding=2, groups=channels, bias=False)
            mask = 1 * torch.ones(1, 1, 5, 5)
            mask[0, 0, 2, 2] = -24
            mask_conv.weight.data = mask.repeat(channels, 1, 1, 1)
            for param in mask_conv.parameters():
                param.requires_grad = False
            self.conv_layers[channels] = mask_conv

        return self.conv_layers[channels]

    def forward(self, img):
        b, c, h, w = img.shape
        mask_conv = self.get_masked_conv(c)
        mask_conv.cuda()
        masked_output = mask_conv(img)
        masked_output = self.relu(masked_output)

        sum_output = masked_output / 24.0

        x = 0.9 * img + 0.25 * sum_output

        return x
