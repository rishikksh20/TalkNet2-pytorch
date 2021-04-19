import torch
import torch.nn as nn
from quartznet import QuartzNet5x5, QuartzNet9x5

class TalkNet2(nn.Module):

    def __init__(self):
        super(TalkNet2, self).__init__()