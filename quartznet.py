import torch
import torch.nn as nn
import torch.nn.functional as F

class SepConv1d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,):
        super(SepConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=(kernel_size - 1) // 2,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)

class ConvBN1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):

      super(ConvBN1d, self).__init__()
      self.conv = nn.Sequential(
          nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                    padding=(kernel_size - 1) // 2),
          nn.BatchNorm1d(out_channels),
          nn.ReLU(),
          nn.Dropout(0.1)
      )

    def forward(self, x):

        return self.conv(x)


class QuartzNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, R=5, dropout=0.1):
        super(QuartzNetBlock, self).__init__()

        model = [SepConv1d(in_channels, out_channels, kernel_size, stride)]

        for i in range(R - 1):
            model += [nn.Dropout(0.1)]
            model += [nn.ReLU()]
            model += [SepConv1d(out_channels, out_channels, kernel_size, stride)]

        self.model = nn.Sequential(*model)

        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, 1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = self.residual(x) + self.model(x)

        return F.relu(x)


class QuartzNet5x5(nn.Module):

    def __init__(self, idim, odim, qdim=256, kernels=[5, 7, 9, 11, 13], embed_dim=64, padding_idx=0):
        super(QuartzNet5x5, self).__init__()

        self.embed = nn.Embedding(idim, embedding_dim=embed_dim, padding_idx=padding_idx)

        self.conv1 = nn.Sequential(
            ConvBN1d(embed_dim, qdim, 3),
            ConvBN1d(qdim, qdim, 3),
            ConvBN1d(qdim, qdim, 3)
        )

        quartznet = []

        for k in kernels:
            quartznet.append(QuartzNetBlock(qdim, qdim, k))
        self.quartznet = nn.Sequential(*quartznet)

        self.conv2 = nn.Sequential(
            ConvBN1d(qdim, qdim * 2, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(qdim * 2, odim, 1)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.conv1(x.transpose(1, 2))
        x = self.quartznet(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class QuartzNet9x5(nn.Module):

    def __init__(self, idim, odim=80, qdim=256, kernels1=[5, 7, 9, 13, 15, 17], kernels2=[21, 23,25], embed_dim=256, padding_idx=0):
        super(QuartzNet9x5, self).__init__()

        self.embed = nn.Embedding(idim, embedding_dim=embed_dim, padding_idx=padding_idx)

        self.conv1 = nn.Sequential(
            ConvBN1d(embed_dim, qdim, 3),
            ConvBN1d(qdim, qdim, 3),
            ConvBN1d(qdim, qdim, 3)
        )

        quartznet = []

        for k in kernels1:
            quartznet.append(QuartzNetBlock(qdim, qdim, k))

        n = qdim * 2
        for k in kernels2:
            quartznet.append(QuartzNetBlock(qdim, n, k))
            qdim = n
        self.quartznet = nn.Sequential(*quartznet)

        self.conv2 = nn.Sequential(
            ConvBN1d(n, n * 2, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(n * 2, odim, 1)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.conv1(x.transpose(1, 2))
        x = self.quartznet(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x




