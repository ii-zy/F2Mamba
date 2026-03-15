import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP: known incorrect sRGB profile.*")
import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,precision_recall_curve


def F1_IoU(pred, target):
    pred = pred.unsqueeze(0).unsqueeze(0)
    target = target.unsqueeze(0).unsqueeze(0)
    if pred.shape[2:] != target.shape[2:]:
        pred = F.interpolate(pred, size=target.shape[2:], mode='nearest')

    target = target.view(-1) 
    pred = pred.view(-1) 
    tp = (target * pred).sum().float()  
    fp = ((1 - target.float()) * pred.float()).sum().float()  
    fn = (target.float() * (1 - pred.float())).sum().float()  
    precision = tp / (tp + fp + 1e-8)  
    recall = tp / (tp + fn + 1e-8)  
    f1 = 2 * precision * recall / (precision + recall + 1e-8) 
    iou = tp / (tp + fp + fn + 1e-8) 
    return f1.item(), iou.item()
def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x


def metrics(pred_dir, mask_dir, dataset_name):
    f1_list, iou_list = [], []
    img_list = os.listdir(pred_dir)
    gt_list = os.listdir(mask_dir)

    for i in tqdm(range(len(img_list))):
        pred_path = os.path.join(pred_dir, img_list[i])
        gt_path = os.path.join(mask_dir, gt_list[i])
        print(pred_path, gt_path)

        image = thresholding(cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE))
        image = image.astype('float') / 255.
        image = torch.from_numpy(image).float()

        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype('float') / 255.
        mask = torch.from_numpy(mask).float()

        f1, iou = F1_IoU(image, mask)
        f1_list.append(f1)
        iou_list.append(iou)
    print(dataset_name)
    print("f1:", np.mean(np.array(f1_list)), "iou:", np.mean(np.array(iou_list)))

    return np.mean(np.array(f1_list)), np.mean(np.array(iou_list))

def calculate_F1_IoU(pred_path, gt_path):
    pred_image = thresholding(cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE))
    pred_image = pred_image.astype('float') / 255.0  
    pred_image = torch.from_numpy(pred_image).float()

    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_image = gt_image.astype('float') / 255.0 
    gt_image = torch.from_numpy(gt_image).float()

    f1, iou = F1_IoU(pred_image, gt_image)

    return f1, iou
def calc_fixed_f1_iou(pred, target):
    pred=pred.unsqueeze(dim=0)
    target=target.squeeze().unsqueeze(dim=0)
    b, n, h, w = pred.size()
    bt, ht, wt = target.size()
    if h != ht or w != wt:
        pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)
    pred = torch.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred, dim=1)
    # F1
    tp = torch.sum(pred_labels[target == 1] == 1).float()
    fp = torch.sum(pred_labels[target == 0] == 1).float()
    fn = torch.sum(pred_labels[target == 1] == 0).float()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    # IoU
    intersection = torch.sum(pred_labels[target == 1] == 1).float()
    union = torch.sum(pred_labels + target >= 1).float()

    iou = intersection / (union + 1e-6)

    return f1_score, iou
def calc_best_f1_auc(y_pred, y_true):

    b, n, h, w = y_pred.size()
    bt, ht, wt = y_true.size()
    if h != ht or w != wt:
        y_pred = F.interpolate(y_pred, size=(ht, wt), mode="bilinear", align_corners=True)

    y_pred = torch.softmax(y_pred, dim=1)[:, 1]
    batchsize = y_true.shape[0]
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        f1_best, auc = 0, 0

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):

            true = y_true[i].flatten()
            true = true.astype(int)
            pred = y_pred[i].flatten()
            precision, recall, thresholds = precision_recall_curve(true, pred)
            try:
                auc += roc_auc_score(true, pred)
            except ValueError:
                pass

            f1_best += max([(2 * p * r) / (p + r + 1e-10) for p, r in
                            zip(precision, recall)])

    return f1_best / batchsize, auc / batchsize


class MyLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(MyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def Softmax_Focal_Loss(self, pred, target):
        target = target.long()
        target = target.squeeze()
        b, c, h, w = pred.size()
        bt, ht, wt = target.size()
        if h != ht or w != wt:
            pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)
        p = F.softmax(pred, dim=1)
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        loss = loss.mean()
        return loss
    def Dice_loss(self, pred, mask):
        smooth = 1e-6
        mask = mask.squeeze(dim=1)
        b, c, h, w = pred.size()
        bt, ht, wt = mask.size()
        if h != ht or w != wt:
            pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)
        pred = torch.softmax(pred, dim=1)[:, 1]
        intersection = torch.sum(pred * mask, dim=(1, 2))
        union = torch.sum(pred, dim=(1, 2)) + torch.sum(mask, dim=(1, 2))
        dice_coefficient = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_coefficient.mean()
        return dice_loss
    def forward(self, pred, mask):
        return self.Dice_loss(pred, mask) + self.Softmax_Focal_Loss(pred, mask)
    

def get_device(cuda_idx):
    cuda_device = os.getenv("CUDA_VISIBLE_DEVICES", str(cuda_idx))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cpu")

    return device

