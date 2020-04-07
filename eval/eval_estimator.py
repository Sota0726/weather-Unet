import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/okada/from_nitta/parm_0.3/sepalated_data_wo-outlier.pkl')
parser.add_argument('--image_root', type=str, default='/mnt/8THDD/takamuro/dataset/photos_usa_2016')
parser.add_argument('--classifer_path', type=str,
                    default='cp/estimator/single/est_res101_flicker-p03th01-WoOutlier_sep-train_aug_pre_loss-l1/est_resnet101_10_step11880.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=16)
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


def make_matricx_img(df, pred, col):
    dir_name = '/mnt/fs2/2019/Takamuro/db/photos_usa_2016/'
    temp_df = df
    temp = pred
    print(temp.shape)
    print('{} gt range is {} ~ {}'.format(col, min(df[col]), max(df[col])))
    print('range is {} ~ {}'.format(np.min(temp), np.max(temp)))
    bins = np.linspace(np.min(temp), np.max(temp), 11)
    print('bins is')
    print(bins)

    from PIL import Image
    img_size = 64
    bin_num = 10
    img_num = 24
    dst = Image.new('RGB', (img_size*bin_num, img_size*img_num))
    for i in range(bin_num):
        temp_df2 = temp_df[(temp_df.clouds>=bins[i])&(temp_df.clouds<bins[i+1])]
        sample_num = min(img_num, len(temp_df2))
        photo_id = temp_df2.sample(n=sample_num)['photo'].tolist()
        # print(photo_id)
        for j, p in enumerate(photo_id):
            im = Image.open(dir_name+p+".jpg")
            im_resize = im.resize((img_size,img_size))
            dst.paste(im_resize, (i*img_size, j*img_size))

    return dst


def plot_hist(col, df, l1, pred):
        gt = df[col].tolist()

        plt.figure()
        plt.hist(gt)
        plt.savefig(os.path.join(save_path, '{}_gt_hist.jpg'.format(col)))

        plt.figure()
        plt.hist(l1)
        plt.savefig(os.path.join(save_path, '{}_l1_hist.jpg'.format(col)))

        plt.figure()
        plt.hist(pred, bins=np.arange(np.min(pred), np.max(pred), 0.25))
        plt.savefig(os.path.join(save_path, '{}_pred_hist.jpg'.format(col)))


if __name__ == '__main__':
    df_ori = pd.read_pickle(args.pkl_path)
    # df = df[:len(df)//2]  # test data
    # df_li = [df[df.condition2 == c _li[i]].sort_values('ent_label')[:100] for i in range(5)]
    # df = pd.concat(df_li)
    save_path = os.path.join('temp-check_est', 'temp_only-est')
    os.makedirs(save_path, exist_ok=True)

    df = df_ori[df_ori['mode'] == 'train']
    cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed']
    # cols = ['temp']

    df_ = df.loc[:, cols].fillna(0)
    df_mean = df_.mean()
    df_std = df_.std()
    # df = df_ori[df_ori['mode'] == 'test']
    df.loc[:, cols] = (df.loc[:, cols].fillna(0) - df_mean) / df_std
    # for col in cols:
    #     tab_img = make_matricx_img(df, df[col].tolist(), col)
    #     tab_img.save('gt_{}.jpg'.format(col))

    print('loaded {} data'.format(len(df)))


    transform = transforms.Compose([
        transforms.Resize((args.input_size,) * 2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FlickrDataLoader(args.image_root, df, cols, transform=transform)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
            )

    # load model
    classifer = torch.load(args.classifer_path)
    classifer.eval()
    classifer.cuda()

    bs = args.batch_size

    av_l1_li = np.empty((0, len(cols)))
    l1_li = np.empty((0, len(cols)))
    pred_li = np.empty((0, len(cols)))

    # vec_li = []
    for i, data in tqdm(enumerate(loader), total=len(df) // bs):
        batch = data[0].to('cuda')
        signals = data[1].to('cuda')
        pred = classifer(batch).detach()

        l1_ = F.l1_loss(pred, signals)
        mse_ = F.mse_loss(pred, signals)
        # l1 = torch.mean(torch.abs(pred - signals), dim=0)
        l1 = torch.abs(pred - signals)

        if len(cols) == 1:
            pred_li = np.append(pred_li, pred.cpu().numpy().reshape(bs, -1))
            l1_li = np.append(l1_li, l1.cpu().numpy().reshape(bs, -1))
            av_l1_li = np.append(l1_li, torch.mean(l1).cpu().numpy().reshape(1, -1))
        else:
            pred_li = np.append(pred_li, pred.cpu().numpy().reshape(bs, -1), axis=0)
            l1_li = np.append(l1_li, l1.cpu().numpy().reshape(bs, -1), axis=0)
            av_l1_li = np.append(l1_li, torch.mean(l1, dim=0).cpu().numpy().reshape(1, -1), axis=0)

    ave_l1 = np.mean(av_l1_li, axis=0)

    with open('l1.pkl', 'wb') as f:
        pickle.dump(l1_li, f)
    with open('pred.pkl', 'wb') as f:
        pickle.dump(pred_li, f)
    # with open('.pkl', 'wb') as f:
    #     pickle.dump(df, f)

    print(cols)
    print(ave_l1)
    print(ave_l1 * df_std)

    # tab_img = make_matricx_img(df, pred_li[:,0])
    if len(cols) == 1:
        tab_img = make_matricx_img(df, pred_li, cols[0])
        tab_img.save(os.path.join(save_path, 'est_{}.jpg'.format(cols[0])))
        plot_hist(cols[0], df, l1_li, pred_li)
    else:
        for i, col in enumerate(cols):
            tab_img = make_matricx_img(df, pred_li[i], col)
            tab_img.save(os.path.join(save_path, 'est_{}.jpg'.format(col)))
            plot_hist(col, df, l1_li[i], pred_li[i])