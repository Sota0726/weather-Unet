import argparse
import pickle
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--image_root', type=str,
                    default='/mnt/fs2/2019/takamuro/db/photos_usa_2016/')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/okada/'
                            'b4_sys/search_parm_new2/parm_0.3/'
                            'sepalated_data.pkl')
parser.add_argument('--output_dir', type=str,
                    default='/mnt/fs2/2019/takamuro/results/c-UNet/'
                            'eval_classifier_flicker/flicker_param-03_th-1_res101_val_imb')
# imb mean imbalanced
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/data2/takamuro/m2/cUNet-Pytorch/cp/'
                            'classifier/resnet101_15.pt')
# parser.add_argument('--classifer_path', type=str, default='cp/classifier/res_aug_5_cls/resnet101_95.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=5)

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
from cunet import Conditional_UNet


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_pickle(args.pkl_path)
    sep_data = df[df['mode'] == 'train']
    print('{} data were loaded'.format(len(df)))

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed', 'rain']
    dataset = FlickrDataLoader(args.image_root, sep_data,
                               cols, transform, class_id=True)

    loader = torch.utils.data.DataLoader(
            dataset,
            sampler=ImbalancedDatasetSampler(dataset),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.num_workers)

    # load model
    classifer = torch.load(args.classifer_path)
    classifer.eval()

    if args.gpu > 0:
        classifer.cuda()

    bs = args.batch_size
    cls_li = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
    s_li = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']

    res_li = []
    for i, data in tqdm(enumerate(loader)):
        batch = data[0].to('cuda')
        c_batch = data[2].to('cuda')
        pred = torch.argmax(classifer(batch).detach(), 1)
        res_li.append(torch.cat([pred.int().cpu().view(bs, -1),
                                c_batch.int().cpu().view(bs, -1)], 1))
    all_res = torch.cat(res_li, 0).numpy()
    y_true, y_pred = (all_res[:, 1], all_res[:, 0])

    table = classification_report(y_true, y_pred)

    print(table)

    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(cls_li)))

    df = pd.DataFrame(data=matrix, index=cls_li, columns=cls_li)
    df.to_pickle(os.path.join(args.output_dir, 'cm.pkl'))

    plot = sns.heatmap(df, square=True, annot=True, fmt='d')

    fig = plot.get_figure()
    fig.savefig(os.path.join(args.output_dir, 'pr_table.png'))
