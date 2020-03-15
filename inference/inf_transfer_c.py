import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--image_root', type=str,
                    default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/db/i2w/sepalated_data.pkl')
parser.add_argument('--output_dir', '-o', type=str,
                    default='/mnt/fs2/2019/takamuro/results/c_UNet/inf/c-flicker/i2w')
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2018/matsuzaki/results/cp/transfer_class/i2w_res_aug_5_cls_n/i2w_res_aug_5_cls_n_e0026.pt')
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/classifier_i2w_for_train_strict_sep/better_resnet101_10.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_classes', type=int, default=5)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset

sys.path.append(os.getcwd())
from dataset import ClassImageLoader
from cunet import Conditional_UNet

if __name__ == '__main__':
    s_li = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']
    os.makedirs(args.output_dir, exist_ok=True)
    sep_data = pd.read_pickle(args.pkl_path)
    sep_data = sep_data['test']
    print('loaded {} data'.format(len(sep_data)))

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ClassImageLoader(paths=sep_data, transform=transform, inf=True)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
            )
    random_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
            )

    # load model
    transfer = Conditional_UNet(num_classes=args.num_classes)
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])

    classifer = torch.load(args.classifer_path)
    classifer.eval()

    # if args.gpu > 0:
    transfer.cuda()
    classifer.cuda()

    bs = args.batch_size

    cls_li = []
    vec_li = []
    for i, (data, rnd) in tqdm(enumerate(zip(loader, random_loader)), total=len(sep_data)//bs):
        batch = data[0].to('cuda')
        r_batch = rnd[0].to('cuda')
        c_batch = rnd[1].to('cuda')
        r_cls = c_batch
        c_batch = F.one_hot(c_batch, args.num_classes).float()
        # r_cls = torch.argmax(classifer(r_batch).detach(), 1)
        out = transfer(batch, c_batch)

        # for check output
        # add "return image, target, self.paths[idx]" to __getitem__ of ClassImageLoader
        path = data[2]
        # for _ in path:
        #     shutil.copy(_, args.output_dir)

        [shutil.copy(_, args.output_dir) for _ in path]
        # _ = [shutil.copy(_, os.path.join(args.output_dir, 'out')) for _ in path]
        [save_image(output, os.path.join(args.output_dir, '{}_'.format(path[j].split('/')[-1].split('.')[0]) + s_li[r_cls[j]] + '.png'), normalize=True) 
         for j, output in enumerate(out)]
        # [save_image(out[(r_cls == j)], os.path.join(args.output_dir, 'out', s_li[j]+'_{}.png'.format(i)), normalize=True) for j in range(5) if len((r_cls == j).nonzero()) != 0]
        # if i>20: exit()
