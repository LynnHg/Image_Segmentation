import torch
import torch.nn as nn


def diceCoeff(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements（预测的元素）
        gt (torch.Tensor):  A list of elements that are to be predicted（待预测的元素）
        eps (float): epsilon to avoid zero division（防止0作为分母的情况发生）
        threshold: threshold for outputs binarization（输出二元值时候使用的阈值）
    Returns:
        float: IoU (Jaccard) score（IoU分数）
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d 激活函数的操作"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    N = gt.size(0)
    pr_flat = pr.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum(gt_flat * pr_flat, dim=1)  # 真正
    fp = torch.sum(pr_flat, dim=1) - tp  # 假正
    fn = torch.sum(gt_flat, dim=1) - tp  # 假负

    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    return score.sum() / N


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeff(y_pr, y_gt, beta=1.,
                        eps=self.eps, threshold=None,
                        activation=self.activation)


if __name__ == "__main__":
    import torch

    pr = torch.Tensor([[[[0.9]],
                        [[0.9]],
                        [[0.9]]],
                       [[[0.9]],
                        [[0.9]],
                        [[0.9]]],
                       [[[0.9]],
                        [[0.9]],
                        [[0.9]]],
                       [[[0.9]],
                        [[0.9]],
                        [[0.9]]]])
    gt = torch.Tensor([[[[1]],
                        [[1]],
                        [[1]]],
                       [[[1]],
                        [[1]],
                        [[1]]],
                       [[[1]],
                        [[1]],
                        [[1]]],
                       [[[1]],
                        [[1]],
                        [[1]]]])

    loss_fn = DiceLoss()
    loss = loss_fn(pr, gt)
    dice = diceCoeff(pr, gt)
    print(dice)
    print(loss)
