import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

from . import box_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        else:
            self.code_weights = None

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        else:
            self.code_weights = None

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)

class CenterNetFocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""
    def __init__(self):
        super(CenterNetFocalLoss, self).__init__()

    def _neg_loss(self, pred, gt):
        """ Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
              pred (batch x c x h x w)
              gt_regr (batch x c x h x w)
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def forward(self, out, target):
        return self._neg_loss(out, target)

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3)).contiguous()
    feat = _gather_feat(feat, ind)
    return feat.contiguous()

class CenterNetRegLoss(nn.Module):
    """Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """

    def __init__(self):
        super(CenterNetRegLoss, self).__init__()

    def _reg_loss(self, regr, gt_regr, mask):
        """ L1 regression loss
            Arguments:
            regr (batch x max_objects x dim)
            gt_regr (batch x max_objects x dim)
            mask (batch x max_objects)
        """
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr).float()
        isnotnan = (~ torch.isnan(gt_regr)).float()
        mask *= isnotnan
        regr = regr * mask
        gt_regr = gt_regr * mask

        loss = torch.abs(regr - gt_regr)
        loss = loss.transpose(2, 0).contiguous()

        loss = torch.sum(loss, dim=2)
        loss = torch.sum(loss, dim=1)

        loss = loss / (num + 1e-4)
        return loss

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        # pred_for_iou = pred.clone().detach()
        loss = self._reg_loss(pred, target, mask)
        return loss
        # return loss, pred_for_iou

class CenterNetRegDimLoss(nn.Module):
    """Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """

    def __init__(self):
        super(CenterNetRegDimLoss, self).__init__()

    def _reg_loss(self, regr, gt_regr, mask):
        """ L1 regression loss
            Arguments:
            regr (batch x max_objects x dim)
            gt_regr (batch x max_objects x dim)
            mask (batch x max_objects)
        """
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr).float()
        isnotnan = (~ torch.isnan(gt_regr)).float()
        mask *= isnotnan
        regr = regr * mask
        gt_regr = gt_regr * mask
        beta = 1.0/9.0
        n = torch.abs(regr - gt_regr)
        loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        loss = loss.transpose(2, 0).contiguous()

        loss = torch.sum(loss, dim=2)
        loss = torch.sum(loss, dim=1)

        loss = loss / (num + 1e-4)
        return loss

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        # pred_for_iou = pred.clone().detach()
        loss = self._reg_loss(pred, target, mask)
        return loss
 
class CornerNetIoULoss(nn.Module):
    """Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """

    def __init__(self):
        super(CornerNetIoULoss, self).__init__()

    def _iou_loss(self, regr, gt_regr, mask, sigma=3):
        """ L1 regression loss
            Arguments:
            regr (batch x max_objects x dim)
            gt_regr (batch x max_objects x dim)
            mask (batch x max_objects)
        """
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr).float()
        isnotnan = (~ torch.isnan(gt_regr)).float()
        mask *= isnotnan
        regr = regr * mask
        gt_regr = gt_regr * mask

        abs_diff = torch.abs(regr - gt_regr)

        abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)

        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + (
                abs_diff - 0.5 / (sigma ** 2)
        ) * (1.0 - abs_diff_lt_1)

        loss = loss.transpose(2, 0).contiguous()

        loss = torch.sum(loss, dim=2)
        loss = torch.sum(loss, dim=1)

        loss = loss / (num + 1e-4)
        return loss
    @torch.no_grad()
    def get_iou(self, pred_reg, gt_reg, coder):
        ious = []
        for i in range(gt_reg.shape[0]):
            pred_bboxes = coder.decode_for_iou(pred_reg[i])
            gt_bboxes = coder.decode_for_iou(gt_reg[i])
            iou = boxes_iou3d_gpu(pred_bboxes, gt_bboxes.to(pred_bboxes.device))
            iou = torch.diag(iou, 0)
            iou[iou > 1] = 0.
            iou[iou < 0] = 0.
            iou = torch.where(torch.isnan(iou), torch.full_like(iou, 0.), iou)
            iou = 2.*(iou - 0.5)
            iou = iou.view(-1, 1)
            ious.append(iou.unsqueeze(0))
        return torch.cat(ious, dim=0)
    def forward(self, output_iou, output_reg, mask, ind, target, coder):
        pred_iou = _transpose_and_gather_feat(output_iou, ind)
        target_iou = self.get_iou(output_reg, target, coder)
        loss = self._iou_loss(pred_iou, target_iou, mask)
        return loss

class CenterNetSmoothRegLoss(nn.Module):
    """Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """

    def __init__(self):
        super(CenterNetSmoothRegLoss, self).__init__()

    def _smooth_reg_loss(self, regr, gt_regr, mask, sigma=3):
        """ L1 regression loss
          Arguments:
            regr (batch x max_objects x dim)
            gt_regr (batch x max_objects x dim)
            mask (batch x max_objects)
        """
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr).float()
        isnotnan = (~ torch.isnan(gt_regr)).float()
        mask *= isnotnan
        regr = regr * mask
        gt_regr = gt_regr * mask

        abs_diff = torch.abs(regr - gt_regr)

        abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)

        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + (
                abs_diff - 0.5 / (sigma ** 2)
        ) * (1.0 - abs_diff_lt_1)

        loss = loss.transpose(2, 0).contiguous()

        loss = torch.sum(loss, dim=2)
        loss = torch.sum(loss, dim=1)

        loss = loss / (num + 1e-4)
        return loss

    def forward(self, output, mask, ind, target, sin_loss):
        assert sin_loss is False
        pred = _transpose_and_gather_feat(output, ind)
        loss = self._smooth_reg_loss(pred, target, mask)
        return loss


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    loss = pos_loss + neg_loss
    return loss

class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma)
        if reduction == 'mean':
            if avg_factor is not None:
                loss_reg = loss_reg.sum() / avg_factor
            else:
                loss_reg = loss_reg.mean()
        return loss_reg

class SEPFocalLoss(nn.Module):
    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SEPFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * separate_sigmoid_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls

def separate_sigmoid_focal_loss(pred,
                                target,
                                weight=None,
                                gamma=2.0,
                                alpha=0.25,
                                reduction='mean',
                                avg_factor=None):
    # pred_sigmoid = pred.sigmoid()
    pred_sigmoid = torch.clamp(pred.sigmoid(), min=1e-4, max=1 - 1e-4)
    target = target.type_as(pred)

    pos_inds = target.eq(1)
    neg_inds = target.lt(1)

    pos_weights = weight[pos_inds]

    pos_pred = pred_sigmoid[pos_inds]
    neg_pred = pred_sigmoid[neg_inds]

    pos_loss = -torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma) * pos_weights * alpha
    neg_loss = -torch.log(1 - neg_pred) * torch.pow(neg_pred, gamma) * (1 - alpha)

    if pos_pred.nelement() == 0:
        loss = neg_loss.sum() / avg_factor
    else:
        loss = pos_loss.sum() / pos_weights.sum() + neg_loss.sum() / avg_factor

    return loss