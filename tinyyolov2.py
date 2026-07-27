import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyYoloV2(nn.Module):
    """
    TinyYoloV2 implementation, which handles cases where the state dict is:
    - pruned
    - variable class count
    - batchnorm fused with conv layers
    """
    def __init__(self, num_classes=1, channels=None, fused=False):
        super().__init__()
        
        # Default to unpruned channels if no channels are passed
        if channels is None:
            channels = [16, 32, 64, 128, 256, 512, 1024, 1024]

        anchors = ((1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),)

        self.register_buffer("anchors", torch.tensor(anchors))
        self.num_classes = num_classes
        self.fused = fused

        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        # If fused=True, convolutions NEED biases to absorb the BatchNorm shift.
        # self.bnX is nn.Identity(), which costs zero computation and we skip
        # the ugly if else statements

        # If fused=False, convolutions do not need biases (BatchNorm handles it).
        
        conv_bias = True if fused else False

        self.conv1 = nn.Conv2d(3, channels[0], 3, 1, 1, bias=conv_bias)
        self.bn1 = nn.Identity() if fused else nn.BatchNorm2d(channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, 1, 1, bias=conv_bias)
        self.bn2 = nn.Identity() if fused else nn.BatchNorm2d(channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, 1, 1, bias=conv_bias)
        self.bn3 = nn.Identity() if fused else nn.BatchNorm2d(channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, 1, 1, bias=conv_bias)
        self.bn4 = nn.Identity() if fused else nn.BatchNorm2d(channels[3])

        self.conv5 = nn.Conv2d(channels[3], channels[4], 3, 1, 1, bias=conv_bias)
        self.bn5 = nn.Identity() if fused else nn.BatchNorm2d(channels[4])

        self.conv6 = nn.Conv2d(channels[4], channels[5], 3, 1, 1, bias=conv_bias)
        self.bn6 = nn.Identity() if fused else nn.BatchNorm2d(channels[5])

        self.conv7 = nn.Conv2d(channels[5], channels[6], 3, 1, 1, bias=conv_bias)
        self.bn7 = nn.Identity() if fused else nn.BatchNorm2d(channels[6])

        self.conv8 = nn.Conv2d(channels[6], channels[7], 3, 1, 1, bias=conv_bias)
        self.bn8 = nn.Identity() if fused else nn.BatchNorm2d(channels[7])

        # Final layer output always has a bias, regardless of fusion
        self.conv9 = nn.Conv2d(channels[7], len(anchors) * (5 + num_classes), 1, 1, 0)

    def forward(self, x, yolo=True):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.pad(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv9(x)
        if yolo:
            nB, _, nH, nW = x.shape

            x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

            anchors = self.anchors.to(dtype=x.dtype, device=x.device)
            range_y, range_x, = torch.meshgrid(
                    torch.arange(nH, dtype=x.dtype, device=x.device),
                    torch.arange(nW, dtype=x.dtype, device=x.device)
            )
            anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

            x = torch.cat([
                (x[:, :, :, :, 0:1].sigmoid() + range_x[None, None, :, :, None]) / nW, #x center
                (x[:, :, :, :, 1:2].sigmoid() + range_y[None, None, :, :, None]) / nH, #y center
                (x[:, :, :, :, 2:3].exp() * anchor_x[None, :, None, None, None])/ nW, # Width
                (x[:, :, :, :, 3:4].exp() * anchor_y[None, :, None, None, None]) /nH, # Height
                x[:, :, :, :, 4:5].sigmoid(), #confidence
                x[:, :, :, :, 5:].softmax(-1),], -1)
        
        return x