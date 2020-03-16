import argparse
import pickle
import os
import sys


import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange, tqdm
from torchvision.utils import save_image

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int)
parser.add_argument('--pkl_path', type=str, default='/mnt/fs2/2018/matsuzaki/results/flickr_data/df_con_less25.pkl')
parser.add_argument('--output_dir', type=str, default='/mnt/fs2/2018/matsuzaki/results/eval/transfer')
parser.add_argument('--image_root', type=str, default='/mnt/fs2/2019/Takamuro/db/photos_usa_2016_outdoor')
#parser.add_argument('--cp_path', type=str, default='/mnt/fs2/2018/matsuzaki/results/cp/out110_res101_e10_less25/out110_res101_e10_less25_e0015.pt')
parser.add_argument('--cp_path', type=str, default='/mnt/fs2/2018/matsuzaki/results/cp/out110_res101_e10_less25/out110_res101_e10_less25_e0017.pt')
parser.add_argument('--classifer_path', type=str, default='cp/classifier/i2w_res101_val_n/resnet101_95.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=6)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset

sys.path.append(os.getcwd())
from dataset import FlickrDataLoader
from sampler import ImbalancedDatasetSampler


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_pickle(args.pkl_path)
    df = df[:len(df)//2]  # test data
    df_li = [df[df.condition2 == c _li[i]].sort_values('ent_label')[:100] for i in range(5)]
    df = pd.concat(df_li)
    print('loaded {} data'.format(len(df)))
    cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed', 'rain']

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FlickrDataLoader(args.image_root, df, cols, transform=transform)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
            )

    # load model
    classifer = torch.load(args.classifer_path)
    classifer.eval()

    if args.gpu > 0:
        classifer.cuda()

    bs = args.batch_size

    cls_li = []
    vec_li = []
    for i, data in tqdm(enumerate(loader), total=len(df)//bs):
        batch = data[0].to('cuda')
        signals = data[1].to('cuda')
        # r_cls = torch.argmax(classifer(r_batch).detach(), 1)
        preds = classifer(batch)
        # for check output
        # [save_image(out[(r_cls == j)], os.path.join(args.output_dir, 'out', s_li[j]+'_{}.png'.format(i)), normalize=True) for j in range(5) if len((r_cls == j).nonzero()) != 0]
        # if i>20: exit()

        c_preds = torch.argmax(classifer(out).detach(), 1)
        cls_li.append(torch.cat([r_cls.int().cpu().view(r_cls.size(0), -1),
                                 c_preds.int().cpu().view(c_preds.size(0), -1)], 1))
    all_res = torch.cat(cls_li, 0).numpy()
    y_true, y_pred = (all_res[:, 0], all_res[:, 1])

