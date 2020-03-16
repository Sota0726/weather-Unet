# to add classifier label to flicker data
import argparse
import os
import sys

import numpy as np
import pandas as pd
from glob import glob
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
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/classifier_i2w_for_train_strict_sep/better_resnet101_10.pt')
# parser.add_argument('--classifer_path', type=str, default='cp/classifier/res_aug_5_cls/resnet101_95.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--image_only', action='store_true')
parser.add_argument('--image_i2w', action='store_true')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torchvision.transforms as transforms
import torch.nn as nn

sys.path.append(os.getcwd())
from dataset import FlickrDataLoader, ClassImageLoader, ImageLoader
# from sampler import ImbalancedDatasetSampler


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
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

    cls_li = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
    if args.image_only:
        paths = glob(os.path.join(args.image_root, '*.png'))
        print('loaded {} data'.format(len(paths)))

        df = pd.DataFrame.from_dict({'paths': [path.split('/')[-1] for path in paths]})
        df['w_condition'] = None
        dataset = ImageLoader(paths=paths, transform=transform)
    elif args.image_i2w:
        cls_li = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']
        df = pd.read_pickle(args.pkl_path)
        df = df['test']
        print('loaded {} data'.format(len(df)))

        df['w_condition'] = None
        dataset = ClassImageLoader(paths=df, transform=transform)
    else:
        df = pd.read_pickle(args.pkl_path)
        cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed', 'rain']
        print('loaded {} data'.format(len(df)))

        df['w_condition'] = None
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

    res_li = []
    print('{} itetates'.format(len(df) / bs))
    for i, data in tqdm(enumerate(loader)):
        batch = data[0].to('cuda')
        ind_batch = data[1].to('cuda')

        pred = torch.argmax(classifer(batch).detach(), 1)

        res_li.append(pred.int().cpu().view(bs, -1))
    all_res = torch.cat(res_li, 0).numpy()
    for i, res in enumerate(all_res):
        df.iat[i, df.columns.get_loc('w_condition')] = cls_li[res[0]]
    print(df)
    df.to_pickle(os.path.join(args.output_dir, 'sep_with_est-label.pkl'))
    # df.to_pickle(os.path.join(args.output_dir, 'check_result.pkl'))
