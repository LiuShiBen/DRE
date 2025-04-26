import torch
import torch.nn.functional as F
from einops import rearrange


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=-1)
    prod = e @ e.transpose(-1, -2)
    res = (e_square.unsqueeze(-2) + e_square.unsqueeze(-1) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[..., range(e.shape[-2]), range(e.shape[-2])] = 0
    return res


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(mat1=x, mat2=y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def loss_fn_kd(scores, target_scores, T=2., return_score=False):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)
    # log_scores_norm = torch.cat([F.log_softmax(score / T, dim=1) for score in scores], dim=1)
    # targets_norm = torch.cat([F.softmax(target_score / T, dim=1) for target_score in target_scores], dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n > target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)
        # log_scores_norm = log_scores_norm[:, :target_scores.size(1)]

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    kd_loss_unnorm = -(targets_norm * log_scores_norm)
    kd_loss_unnorm = kd_loss_unnorm.sum(dim=1)  # --> sum over classes
    kd_loss_unnorm = kd_loss_unnorm.mean()  # --> average over batch

    # normalize
    kd_loss = kd_loss_unnorm * T ** 2

    if not return_score:
        return kd_loss
    else:
        return kd_loss, targets_norm.max(dim=1)[0].mean()


def loss_fn_rd(student, teacher):
    t_d = pdist(teacher, squared=False)
    mean_td = t_d[t_d > 0].mean()
    t_d = t_d / mean_td

    d = pdist(student, squared=False)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    loss = F.smooth_l1_loss(d, t_d, reduction='mean')
    return loss


def loss_fn_div(feature, k):
    feature = rearrange(F.normalize(feature), '(n k) c -> n k c', k=k)
    M = feature @ feature.transpose(1, 2)
    return (M.sum(dim=(1, 2)) - k).mean() / (k * (k - 1))
