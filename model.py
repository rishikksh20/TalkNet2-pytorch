import torch
import torch.nn as nn
from quartznet import QuartzNet5x5, QuartzNet9x5


class GraphemeDuration(nn.Module):

    def __init__(self):
        super(GraphemeDuration, self).__init__()

        self.predictor = QuartzNet5x5(1, 32)
        self.projection = nn.Linear(32, 1)

    def forward(self, x):

        out = self.predictor(x)
        out = self.projection(out.transpose(-2, -1))

        return out
# Work remain : Gaussian embedding
class PitchPredictor(nn.Module):

    def __init__(self):
        super(PitchPredictor, self).__init__()

        self.predictor = QuartzNet5x5(1, 32)
        self.uv_out = nn.Linear(32, 2)
        self.value_out = nn.Linear(32, 1)

    def forward(self, x):

        out = self.predictor(x)
        uv = self.uv_out(out.transpose(-2, -1))
        value = self.value_out(out.transpose(-2, -1))

        return uv, value

# Work remain : Gaussian embedding and Postnet
class TalkNet2(nn.Module):

    def __init__(self):
        super(TalkNet2, self).__init__()

        self.generator = QuartzNet9x5(1, 80)


    def forward(self, x):

        return self.generator(x)