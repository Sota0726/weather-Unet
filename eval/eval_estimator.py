import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/okada/from_nitta/parm_0.3/for_transfer-est_training.pkl')
parser.add_argument('--image_root', type=str, default='/mnt/8THDD/takamuro/dataset/photos_usa_2016')
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/estimator/est_res101_flicker-p03th1_sep-val/est_resnet101_15_step17760.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=6)
args = parser.parse_args()
# GPU Setting
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append(os.getcwd())
from dataset import FlickrDataLoader


if __name__ == '__main__':
    df = pd.read_pickle(args.pkl_path)
    # df = df[:len(df)//2]  # test data
    # df_li = [df[df.condition2 == c _li[i]].sort_values('ent_label')[:100] for i in range(5)]
    # df = pd.concat(df_li)
    df = df[df['mode'] == 'test']
    print('loaded {} data'.format(len(df)))
    cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed', 'rain']

    transform = transforms.Compose([
        transforms.Resize((args.input_size,) * 2),
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
    classifer.cuda()

    bs = args.batch_size

    l1_li = np.empty((0, 6))
    # vec_li = []
    for i, data in tqdm(enumerate(loader), total=len(df) // bs):
        batch = data[0].to('cuda')
        signals = data[1].to('cuda')
        pred = classifer(batch).detach()

        l1_ = F.l1_loss(pred, signals)
        mse_ = F.mse_loss(pred, signals)

        l1 = torch.mean(torch.abs(pred - signals), dim=0)
        l1_li = np.append(l1_li, l1.cpu().numpy().reshape(1, -1), axis=0)
    ave_l1 = np.mean(l1_li, axis=0)

    df_ = df.loc[:, cols]
    df_meam = df_.mean().to_list()
    df_std = df_.std().to_list()

    print(cols)
    print(ave_l1)
    print(ave_l1 * df_std)
