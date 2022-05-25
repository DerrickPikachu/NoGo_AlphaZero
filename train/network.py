import torch.nn as nn
import torch

class AutoPaddingConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, is_extend=False, out_channels=None):
        super(AutoPaddingConv2d, self).__init__()
        if out_channels is None:
            out_channels = in_channels if not is_extend else in_channels * 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, \
            bias=False, padding=kernel_size - 2)
    
    def forward(self, x):
        return self.conv(x)
    

class BasicBlock(nn.Module):
    def __init__(self, in_channels, is_extend):
        super(BasicBlock, self).__init__()
        if not is_extend:
            out_channels = in_channels
            self.identity = nn.Identity()
        else:
            out_channels = in_channels * 2
            self.identity = nn.Conv2d(in_channels, out_channels, 1)

        self.bone = nn.Sequential(
            AutoPaddingConv2d(in_channels, 3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            AutoPaddingConv2d(in_channels, 3, is_extend),
            nn.BatchNorm2d(out_channels),
        )

        self.out_activation = nn.ReLU()

    def forward(self, x):
        identity_mapping = self.identity(torch.clone(x))
        conv_result = self.bone(x)
        shortcut_sum = conv_result + identity_mapping
        return self.out_activation(shortcut_sum)
    

class BottleneckBlock(nn.Module):
    def __init__(self):
        super(BottleneckBlock, self).__init__()
        

class AlphaZeroResnet(nn.Module):
    def __init__(self, in_channels, num_blocks, hidden_channels=256, input_size=(19, 19)):
        super(AlphaZeroResnet, self).__init__()
        self.conv_head = nn.Sequential(
            AutoPaddingConv2d(in_channels, 3, out_channels=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.resnet = nn.ModuleList([
            *[BasicBlock(hidden_channels, is_extend=False)
                for _ in range(num_blocks)]
        ])
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_size[0] * input_size[1] * 2,\
                input_size[0] * input_size[1])
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_size[0] * input_size[1], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.conv_head(x)
        for block in self.resnet:
            x = block(x)
        feature = x
        policy = self.policy_head(torch.clone(feature))
        value = self.value_head(torch.clone(feature))
        return policy, value


if __name__ == "__main__":
    resnet = AlphaZeroResnet(3, 5)
    print(resnet)