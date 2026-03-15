from Datasets import MyDataset
from torch.utils.data import DataLoader
import argparse
import logging as logger
import torch.nn as nn
import shutil
import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from models.F2_Mamba import F2Mamba_Loc
from metric import calc_fixed_f1_iou
import datetime

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s] %(message)s',
                   datefmt='%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--weight_path', type=str, default='./F^2Mamba_Localization_weights.pth',
                    help='weight path of trained model')
parser.add_argument('--input_size', type=int, default=512, help='size of resized input')
parser.add_argument('--gt_ratio', type=int, default=1, help='resolution of input / output')
parser.add_argument('--save_result', type=bool, default=True, help='save test results')
parser.add_argument('--test_bs', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')

args = parser.parse_args()
logger.info(args)

date_now = datetime.datetime.now()
date_now = 'Prediction_%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
args.out_dir = date_now
device = torch.device('cuda:{}'.format(args.gpu))


class IML_infer(nn.Module):
    def __init__(self):
        super(IML_infer, self).__init__()
        self.cur_net = F2Mamba_Loc().to(device)
        self.load(self.cur_net, args.weight_path)

    def process(self, Ii):
        with torch.no_grad():
            Fo = self.cur_net(Ii)
        return Fo

    def load(self, model, path):
        weights = torch.load(path, map_location=torch.device('cpu'))['model_state_dict']
        model_state_dict = model.state_dict()

        loaded_layers = []
        missing_layers = []
        mismatched_shapes = []

        for name, param in weights.items():
            if name in model_state_dict:
                if param.shape == model_state_dict[name].shape:
                    model_state_dict[name].copy_(param)  
                    loaded_layers.append(name)
                else:
                    mismatched_shapes.append(name)
            else:
                missing_layers.append(name)

        if loaded_layers:
            print('Successfully loaded all layers')

        if mismatched_shapes:
            logger.warning(f"The following layers have mismatched shapes: {', '.join(mismatched_shapes)}")

        if missing_layers:
            logger.warning(f"The following layers are missing in the model: {', '.join(missing_layers)}")

        if not mismatched_shapes and not missing_layers:
            logger.info("All layers have been successfully loaded!")


class IML_Loc():
    def __init__(self):
        self.F2Mamba = IML_infer().to(device)
        self.test_npy_list = [
    
            ('Columbia_160.npy','Columbia_160'),
            ('DSO_100.npy','DSO_100'),
            ('coverage_100.npy', 'coverage_100'),
            ('in_wild_201.npy','in_wild_201'),
            ('CASIA_v1_920.npy','CASIA_v1_920')

           
        ]
        self.test_file_list = []
        for item in self.test_npy_list:
            self.test_file_tmp = np.load('Train_list/' + item[0])
            self.test_file_list.append(self.test_file_tmp)
        self.test_bs = args.test_bs

        for idx, file_list in enumerate(self.test_file_list):
            logger.info('Test on %s (#%d).' % (self.test_npy_list[idx][0], len(file_list)))

    def test(self):
        tmp_F1 = []
        tmp_IOU = []
        result_file_path = os.path.join(args.out_dir, 'result.txt')
        os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
        with open(result_file_path, 'a') as result_file:  
            for idx in range(len(self.test_file_list)):
                P_F1, P_IOU = IML_test(self.F2Mamba, bs=self.test_bs,
                                              test_file=self.test_file_list[idx],
                                              test_set_name=self.test_npy_list[idx][1])
                tmp_IOU.append(P_IOU)
                tmp_F1.append(P_F1)
                result_str = '%s(#%d): F1:%5.4f, PIOU:%5.4f\n' % (
                    self.test_npy_list[idx][1], len(self.test_file_list[idx]), P_F1, P_IOU
                )
                result_file.write(result_str)
                result_file.flush()


def IML_test(model, bs=1, test_file=None, test_set_name=None):
    test_num = len(test_file)
    test_dataset = MyDataset(test_num, test_file, choice='test', input_size=args.input_size, gt_ratio=args.gt_ratio)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=min(48, 2), shuffle=False)

    model.eval()
    f1, iou = [], []

    save_dir = os.path.join(args.out_dir, test_set_name)
    os.makedirs(save_dir, exist_ok=True) 

    for items in test_loader:
        Ii, Mg, Hg, Wg = (item.to(device) for item in items[:-1])
        filename = items[-1]
        Mo = model.process(Ii)  # [B, 2, 128, 128]

        if args.save_result:
            Hg, Wg = Hg.cpu().numpy(), Wg.cpu().numpy()
            for i in range(Mo.shape[0]):
                Mo_softmax = F.softmax(Mo[i], dim=0)
                Mo_argmax = torch.argmax(Mo_softmax, dim=0).cpu().numpy()
                Mo_normalized = (Mo_argmax * 255).astype(np.uint8)
                Mo_resized = Image.fromarray(Mo_normalized).resize((Wg[i], Hg[i]), Image.NEAREST)
                save_path = os.path.join(save_dir, filename[i].split('.')[-2] + '.png')
                Mo_resized.save(save_path)

        for i in range(Mo.shape[0]):
            fixed_f1, iou_score = calc_fixed_f1_iou(Mo[i], Mg[i])
            f1.append(fixed_f1.cpu())
            iou.append(iou_score.cpu())

    Pixel_F1 = np.mean(f1)
    Pixel_IOU = np.mean(iou)
    return Pixel_F1, Pixel_IOU


if __name__ == '__main__':
    model = IML_Loc()
    model.test()
