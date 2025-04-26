import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange


def softmax_hard(y, tau=1, dim=-1):
    y_soft = (y / tau).softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft


class PatchNet(nn.Module):
    def __init__(self, K=3):
        super(PatchNet, self).__init__()
        self.K = K

        self.scorer = nn.Sequential(
            nn.Conv2d(2048, 4096, kernel_size=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(True),
            # nn.Conv2d(4096, 1, kernel_size=1),
            nn.Conv2d(4096, self.K, kernel_size=1)
        )

    def forward(self, x):
        score = self.scorer(x)
        score_flat = rearrange(score, 'n c h w -> n c (h w)')
        sample = F.gumbel_softmax(score_flat, hard=True)
        return sample.reshape_as(score)
        # score_flat = rearrange(score, 'n 1 h w -> n 1 (h w)')
        # # sample gumbels
        # score_flat -= torch.empty_like(score_flat, memory_format=torch.legacy_contiguous_format).exponential_().log()
        # # gumbel top-k
        # one_hot = torch.zeros_like(score_flat)
        # k_hot = []
        # for _ in range(self.K):
        #     # mask = (1 - one_hot).float().clamp(1e-20)
        #     # score_flat += torch.log(mask)
        #     score_flat[one_hot == 1] = float('-inf')
        #     one_hot = softmax_hard(score_flat)
        #     k_hot.append(one_hot)
        # return torch.cat(k_hot, dim=1).reshape(score.shape[0], -1, *score.shape[2:])
