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
parser.add_argument('--gpu', type=str)
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/okada/from_nitta/parm_0.3/50test_high-consis_10images-each-con2.pkl')
parser.add_argument('--image_root', type=str, default='/mnt/fs2/2019/Takamuro/db/photos_usa_2016')
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/transfer/'
                    'cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500/cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500_e0023_s322000.pt')
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/classifier/cls_res101_i2w_sep-val_aug_20200408/resnet101_epoch20_step77847.pt')
parser.add_argument('--estimator_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/estimator/'
                            'est_res101_flicker-p03th01-WoOutlier_sep-val_aug_pre_loss-mse-reduction-none-grad-all-1/est_resnet101_20_step22680.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=6)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset

sys.path.append(os.getcwd())
from dataset import FlickrDataLoader
from cunet import Conditional_UNet


def eval_classifier(batch, c_batch, r_label, cls_li):
    for j in range(bs):
        with torch.no_grad():
            ref_sig_expand = torch.cat([c_batch[j]] * bs).view(-1, len(cols))
            ref_label_expand = torch.cat([r_label[j]] * bs).view(-1, len(c_li))
            out = transfer(batch, ref_sig_expand)

            c_preds = torch.argmax(classifer(out), 1)
            r_cls = torch.argmax(ref_label_expand, 1)
            # [save_image(_, os.path.join(save_path, 'out', '{}_{}.jpg'.format(r_photos[j], b_photos[k])), normalize=True)
            #     for k, _ in enumerate(out)]

        cls_li.append(torch.cat([r_cls.int().cpu().view(r_cls.size(0),-1), c_preds.int().cpu().view(c_preds.size(0),-1)], 1))
    return cls_li


def eval_estimator(batch, b_sig, r_sig, l1_li, l1_li_):
    for j in range(bs):
        with torch.no_grad():
            ref_sig_expand = torch.cat([r_sig[j]] * bs).view(-1, len(cols))
            out = transfer(batch, ref_sig_expand)

            pred_sig = estimator(out)
            # l1 = torch.abs(pred_sig - ref_sig_expand)
            l1 = pred_sig - ref_sig_expand
            l1_ = b_sig - ref_sig_expand
            l1_li = np.append(l1_li, torch.mean(l1, dim=0).cpu().numpy().reshape(1, -1), axis=0)
            l1_li_ = np.append(l1_li_, torch.mean(l1_, dim=0).cpu().numpy().reshape(1, -1), axis=0)
            # [save_image(output, os.path.join(args.output_dir,
            #     '{}_t-{}_r-{}.jpg'.format('gt', b_photos[j], r_photos[i])), normalize=True)
            #     for j, output in enumerate(out)]
    return l1_li, l1_li_


if __name__ == '__main__':
    s_li = ['sunny', 'cloudy', 'rain', 'foggy', 'snow']
    c_li = ['Clear', 'Clouds', 'Rain', 'Mist', 'Snow']

    save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/results/eval_est_transfer',
                             args.cp_path.split('/')[-2],
                             args.cp_path.split('/')[-1].split('_')[-2])

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'out'), exist_ok=True)
    cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed']

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    df = pd.read_pickle(args.pkl_path)

    temp = pd.read_pickle('/mnt/fs2/2019/okada/from_nitta/parm_0.3/sepalated_data_wo-outlier.pkl')
    df_ = temp.loc[:, cols].fillna(0)
    df_mean = df_.mean()
    df_std = df_.std()

    df.loc[:, cols] = (df.loc[:, cols].fillna(0) - df_mean) / df_std

    df_sep = df[df['mode'] == 'test']

    print('loaded {} data'.format(len(df_sep)))

    dataset = FlickrDataLoader(args.image_root, df, cols, transform=transform, class_id=True)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
            )
    random_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            )

    # load model
    transfer = Conditional_UNet(num_classes=len(cols))
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])

    # classifer = torch.load(args.classifer_path)
    # classifer.eval()

    estimator = torch.load(args.estimator_path)
    estimator.eval()

    transfer.cuda()
    # classifer.cuda()
    estimator.cuda()

    bs = args.batch_size

    cls_li = []
    l1_li = np.empty((0, len(cols)))
    l1_li_ = np.empty((0, len(cols)))

    for i, (data, rnd) in tqdm(enumerate(zip(loader, random_loader)), total=len(df)//bs):
        batch = data[0].to('cuda')
        b_sig = data[1].to('cuda')
        b_label = data[2]
        # r_batch  = rnd[0].to('cuda')
        r_sig = rnd[1].to('cuda')
        r_label = torch.eye(len(c_li))[rnd[2]].to('cuda')
        # r_cls = torch.argmax(classifer(r_batch).detach(), 1)
        b_photos = data[3]
        r_photos = rnd[3]
        pred_ = estimator(batch).detach()
        # cls_li = eval_classifier(batch, r_sig, r_label, cls_li)
        l1_li, l1_li_ = eval_estimator(batch, b_sig, r_sig, l1_li, l1_li_)

    ave_l1 = np.mean(l1_li, axis=0)
    std_l1 = np.std(l1_li, axis=0)

    ave_l1_ = np.mean(l1_li_, axis=0)
    std_l1_ = np.std(l1_li_, axis=0)

    print(cols)
    print('l1')
    print(ave_l1)
    print((ave_l1 * df_std))
    print('l1 std')
    print(std_l1)
    print((std_l1 * df_std))

    print()
    print(cols)
    print('l1_')
    print(ave_l1_)
    print((ave_l1_ * df_std))
    print('l1 std_')
    print(std_l1_)
    print((std_l1_ * df_std))
    # print('mse')
    # print(ave_mse)

    # all_res = torch.cat(cls_li, 0).numpy()
    # y_true, y_pred = (all_res[:, 0], all_res[:, 1])

    # table = classification_report(y_true, y_pred)

    # print(table)

    # matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(s_li)))
    # df = pd.DataFrame(data=matrix, index=s_li, columns=s_li)
    # df.to_pickle(os.path.join(save_path, 'cm.pkl'))

    # plot = sns.heatmap(df, square=True, annot=True, fmt='d')
    # fig = plot.get_figure()
    # fig.savefig(os.path.join(save_path, 'pr_table.png'))
