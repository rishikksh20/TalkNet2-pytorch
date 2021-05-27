
import torch.nn as nn
from torch.nn import functional as F
from utils import get_mask_from_lengths
from embedding import GaussianEmbedding
from quartznet import QuartzNet5x5, QuartzNet9x5
from module import MaskedInstanceNorm1d, StyleResidual

# Work remains: Apply masking on Conv1d and Postnet

class GraphemeDuration(nn.Module):

    def __init__(self, idim, embed_dim=64, padding_idx=0):
        super(GraphemeDuration, self).__init__()
        self.embed = nn.Embedding(idim, embedding_dim=embed_dim, padding_idx=padding_idx)
        self.predictor = QuartzNet5x5(embed_dim, 32)
        self.projection = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, text, text_len, is_mask=True):
        x, x_len = self.embed(text).transpose(1, 2), text_len
        if is_mask:
            mask = get_mask_from_lengths(x_len)
        else:
            mask = None
        out = self.predictor(x, mask)
        out = self.projection(out).squeeze(1)

        return out

    @staticmethod
    def _metrics(true_durs, true_text_len, pred_durs):
        loss = F.mse_loss(pred_durs, (true_durs + 1).float().log(), reduction='none')
        mask = get_mask_from_lengths(true_text_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()

        durs_pred = pred_durs.exp() - 1
        durs_pred[durs_pred < 0.0] = 0.0
        durs_pred = durs_pred.round().long()

        acc = ((true_durs == durs_pred) * mask).sum().float() / mask.sum() * 100
        acc_dist_1 = (((true_durs - durs_pred).abs() <= 1) * mask).sum().float() / mask.sum() * 100
        acc_dist_3 = (((true_durs - durs_pred).abs() <= 3) * mask).sum().float() / mask.sum() * 100

        return loss, acc, acc_dist_1, acc_dist_3


class PitchPredictor(nn.Module):

    def __init__(self, idim,  embed_dim=64):
        super(PitchPredictor, self).__init__()
        self.embed = GaussianEmbedding(idim, embed_dim)
        self.predictor = QuartzNet5x5(embed_dim, 32)
        self.sil_proj = nn.Conv1d(32, 1, kernel_size=1)
        self.body_proj = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, text, durs, is_mask=True):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        if is_mask:
            mask = get_mask_from_lengths(x_len)
        else:
            mask = None
        out = self.predictor(x, mask)
        uv = self.sil_proj(out).squeeze(1)
        value = self.body_proj(out).squeeze(1)

        return uv, value

    def _metrics(self, true_f0, true_f0_mask, pred_f0_sil, pred_f0_body):
        sil_mask = true_f0 < 1e-5
        sil_gt = sil_mask.long()
        sil_loss = F.binary_cross_entropy_with_logits(input=pred_f0_sil, target=sil_gt.float(), reduction='none', )
        sil_loss *= true_f0_mask.type_as(sil_loss)
        sil_loss = sil_loss.sum() / true_f0_mask.sum()
        sil_acc = ((torch.sigmoid(pred_f0_sil) > 0.5).long() == sil_gt).float()  # noqa
        sil_acc *= true_f0_mask.type_as(sil_acc)
        sil_acc = sil_acc.sum() / true_f0_mask.sum()

        body_mse = F.mse_loss(pred_f0_body, (true_f0 - self.f0_mean) / self.f0_std, reduction='none')
        body_mask = ~sil_mask
        body_mse *= body_mask.type_as(body_mse)  # noqa
        body_mse = body_mse.sum() / body_mask.sum()  # noqa
        body_mae = ((pred_f0_body * self.f0_std + self.f0_mean) - true_f0).abs()
        body_mae *= body_mask.type_as(body_mae)  # noqa
        body_mae = body_mae.sum() / body_mask.sum()  # noqa

        loss = sil_loss + body_mse

        return loss, sil_acc, body_mae


class TalkNet2(nn.Module):

    def __init__(self, idim, odim=80, embed_dim=256):
        super(TalkNet2, self).__init__()
        self.embed = GaussianEmbedding(idim, embed_dim)
        self.norm_f0 = MaskedInstanceNorm1d(1)
        self.res_f0 = StyleResidual(embed_dim, 1, kernel_size=3)

        self.generator = QuartzNet9x5(embed_dim, odim)


    def forward(self, text, durs, f0, is_mask=True):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        f0, f0_mask = f0.clone(), f0 > 0.0
        f0 = self.norm_f0(f0.unsqueeze(1), f0_mask)
        f0[~f0_mask.unsqueeze(1)] = 0.0
        x = self.res_f0(x, f0)
        if is_mask:
            mask = get_mask_from_lengths(x_len)
        else:
            mask = None
        return self.generator(x, mask)

    @staticmethod
    def _metrics(true_mel, true_mel_len, pred_mel):
        loss = F.mse_loss(pred_mel, true_mel, reduction='none').mean(dim=-2)
        mask = get_mask_from_lengths(true_mel_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()
        return loss