import torch
import torch.nn as nn


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionBlockV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i) for i in range(num_kernels)]
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply each kernel and average their results
        outputs = [kernel(x) for kernel in self.kernels]
        return torch.stack(outputs, dim=-1).mean(-1)


class InceptionBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionBlockV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1])
                if i % 2 == 0 else nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0])
                for i in range(num_kernels // 2)
            ]
        )
        self.kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply each kernel and average their results
        outputs = [kernel(x) for kernel in self.kernels]
        return torch.stack(outputs, dim=-1).mean(-1)
