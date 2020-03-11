# to add classifier label to flicker data
import argparse
import os
import sys

import numpy as np
import pandas as pd
# from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--image_root', type=str,
                    default='/mnt/fs2/2019/takamuro/db/photos_usa_2016/')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/okada/from_nitta/parm_0.3/sepalated_data.pkl')
parser.add_argument('--output_dir', type=str,
                    default='/mnt/fs2/2019/okada/from_nitta/parm_0.3/')
# imb mean imbalanced
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/data2/takamuro/m2/cUNet-Pytorch/cp/classifier/classifier_i2w_strict/better_resnet101_10.pt')
# parser.add_argument('--classifer_path', type=str, default='cp/classifier/res_aug_5_cls/resnet101_95.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=46)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=5)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torchvision.transforms as transforms
import torch.nn as nn

sys.path.append(os.getcwd())
from dataset import FlickrDataLoader
# from sampler import ImbalancedDatasetSampler


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_pickle(args.pkl_path)

    # load model
    classifer = torch.load(args.classifer_path)
    classifer = nn.Sequential(
                        classifer,
                        nn.Softmax(dim=1)
                    )
    classifer.eval()

    if args.gpu > 0:
        classifer.cuda()

    transform = transforms.Compose([
            transforms.Resize((args.input_size,)*2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed', 'rain']
    df['w_condition'] = None
    print('loaded {} data'.format(len(df)))

    dataset = FlickrDataLoader(args.image_root, df, cols, transform, class_id=False)

    loader = torch.utils.data.DataLoader(
            dataset,
            # sampler=ImbalancedDatasetSampler(dataset),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers)

    bs = args.batch_size

    if len(df) % bs != 0:
        print('not divisible by batch size')
        sys.exit()

    cls_li = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
    res_li = []
    print('{} itetates'.format(len(df) / bs))
    for i, data in tqdm(enumerate(loader)):
        batch = data[0].to('cuda')
        c_batch = data[1].to('cuda')
        ind_batch = data[2].to('cuda')

        pred = torch.argmax(classifer(batch).detach(), 1)

        res_li.append(torch.cat([pred.int().cpu().view(bs, -1),
                                ind_batch.int().cpu().view(bs, -1)], 1))
    all_res = torch.cat(res_li, 0).numpy()

    for res in all_res:
        df.iat[res[1], df.columns.get_loc('w_condition')] = cls_li[res[0]]
    print(df['w_condition'])
    df.to_pickle(os.path.join(args.output_dir, 'sep_with_est-label.pkl'))
