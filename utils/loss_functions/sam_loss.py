from pyexpat import model
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class DC_and_BCE_loss(nn.Module):
    def __init__(self, classes=2, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.ce =  CrossEntropyLoss()
        self.dc = DiceLoss(classes)
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        print(f"Calculating DC_and_CE_loss_Multi_Class...")
        print(f"net_output.shape: {net_output.shape}")
        print(f"target.shape: {target.shape}")

        # shape: (B, C, H, W)
        low_res_logits = net_output['low_res_logits']
        print(f"low_res_logits.shape: {low_res_logits.shape}")

        # shape: (B, H, W)
        if len(target.shape) == 4:
            target = target[:, 0, :, :]
        print(f"Reshaped target.shape: {target[:].shape}")

        loss_ce = self.ce(low_res_logits, target[:].long())
        loss_dice = self.dc(low_res_logits, target, softmax=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss

class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, net_output, target, weight=None, sigmoid=False):
        if sigmoid:
            net_output = torch.sigmoid(net_output) # b 1 h w
        assert net_output.size() == target.size(), 'predict {} & target {} shape do not match'.format(net_output.size(), target.size())
        dice_loss = self._dice_loss(net_output[:, 0], target[:, 0])
        return dice_loss

class Mask_DC_and_BCE_loss(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_loss, self).__init__()

        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits']
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.dc(low_res_logits, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss

class Mask_DC_and_BCE_lossV2(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_lossV2, self).__init__()

        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.dc(low_res_logits, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss

class Mask_BCE_loss(nn.Module):
    def __init__(self, pos_weight):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_BCE_loss, self).__init__()

        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits'] 
        loss = self.ce(low_res_logits, target)
        return loss

def get_criterion(modelname='SAM', opt=None):
    device = torch.device(opt.device)
    pos_weight = torch.ones([1]).cuda(device=device)*2
    if modelname == "SAMed":
        criterion = DC_and_BCE_loss(classes=opt.classes)
    elif modelname == "MSA":
        criterion = Mask_BCE_loss(pos_weight=pos_weight)
    elif modelname == "XMemSAM" or modelname == "MemSAM":
        criterion = Mask_DC_and_BCE_lossV2(pos_weight=pos_weight)
    elif opt.task == "Task2":
        print(f"Using multi-class Dice + Cross-Entropy loss for {opt.classes} classes.")
        criterion = Mask_DC_and_CE_loss_Multi_Class(n_classes=opt.classes)
    else:
        criterion = Mask_DC_and_BCE_loss(pos_weight=pos_weight)
    return criterion


class Mask_DC_and_CE_loss_Multi_Class(nn.Module):
    """
    Modification based on DC_and_BCE_loss.
    A loss function for multi-class video segmentation.
    Combines Dice Loss and Cross-Entropy Loss.
    """

    def __init__(self, n_classes, dice_weight=0.8, ignore_index=255):
        super(Mask_DC_and_CE_loss_Multi_Class, self).__init__()
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dc = DiceLoss(n_classes=n_classes)

    def forward(self, net_output, target):
        """
        Args:
            net_output (Tensor): The model's prediction, shape (B, T, C, H, W)
                                 where C is the number of classes.
            target (Tensor): The ground truth mask, shape (B, T, H, W)
        """
        print(f"Calculating Mask_DC_and_CE_loss_Multi_Class...")
        print(f"net_output.shape: {net_output.shape}")
        print(f"target.shape: {target.shape}")
        # Reshape for loss calculation
        # temporal: T, class number: C
        # (B, T, C, H, W) -> (B*T, C, H, W)
        b, t, c, h, w = net_output.shape
        net_output_reshaped = net_output.view(b * t, c, h, w)
        print(f"net_output_reshaped.shape: {net_output_reshaped.shape}")

        # (B, T, H, W) -> (B*T, H, W)
        target_reshaped = target.view(b * t, h, w)
        print(f"target_reshaped.shape: {target_reshaped.shape}")

        # Calculate Cross-Entropy Loss
        loss_ce = self.ce(net_output_reshaped, target_reshaped.long())

        # Calculate Dice Loss
        # The DiceLoss expects a one-hot encoded target, which it handles internally.
        # It also expects softmax to be applied to the network output.
        loss_dice = self.dc(net_output_reshaped, target_reshaped, softmax=True)

        # Combine the losses
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss

