import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil
from torchvision.utils import save_image

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--image_root', type=str,
                    default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/db/i2w/sepalated_data.pkl')
parser.add_argument('--output_dir', '-o', type=str,
                    default='/mnt/data2/takamuro/temp')
parser.add_argument('--cp_path', type=str,
                    # default='/mnt/fs2/2018/matsuzaki/results/cp/transfer_class/i2w_res_aug_5_cls_n/i2w_res_aug_5_cls_n_e0026.pt')
                    default='/mnt/data2/takamuro/m2/cUNet-Pytorch/cp/transfer/w-c-12w-res101_img-i2w_train_D2T1_fixbug/w-c-12w-res101_img-i2w_train_D2T1_fixbug_e0039_s145000.pt')
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/classifier_i2w_for_train_strict_sep/better_resnet101_10.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=8)
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
from dataset import ClassImageLoader
from sampler import ImbalancedDatasetSampler
from cunet import Conditional_UNet
from disc import SNDisc

if __name__ == '__main__':
    s_li = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_pickle(args.pkl_path)
    df = df['test']
    ind_li = []
    for s in s_li:
        ind_li.append([i for i, c in enumerate(p.split('/')[-2] for p in df) if c == s])
    ind_li = np.concatenate([ind[:91] for ind in ind_li])
    print(ind_li.shape)
    df = [df[i] for i in ind_li]
    print('loaded {} data'.format(len(df)))

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ClassImageLoader(paths=df, transform=transform)

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
    # transfer = Conditional_UNet(num_classes=args.num_classes)
    dis = SNDisc(num_classes=args.num_classes)
    sd = torch.load(args.cp_path)
    dis.load_state_dict(sd['discriminator'])

    transfer = Conditional_UNet(num_classes=args.num_classes)
    transfer.load_state_dict(sd['inference'])

    classifer = torch.load(args.classifer_path)
    classifer = nn.Sequential(
                        classifer,
                        nn.Softmax(dim=1)
                    )
    classifer.eval()

    if args.gpu > 0:
        transfer.cuda()
        dis.cuda()
        classifer.cuda()

    bs = args.batch_size

    cls_li = []
    vec_li = []
    for i, (data, rnd) in tqdm(enumerate(zip(loader, random_loader)), total=len(df) // bs):
        batch = data[0].to('cuda')
        r_batch = rnd[0].to('cuda')
        c_batch = rnd[1].to('cuda')
        r_cls = c_batch
        c_batch = F.one_hot(c_batch, args.num_classes).float()
        pred_label = classifer(r_batch).detach()
        print('est label')
        print(pred_label)
        # r_cls = torch.argmax(classifer(r_batch).detach(), 1)
        # out = transfer(batch, c_batch)
        out_t = transfer(batch, pred_label)
        out_d = dis(out_t.detach(), pred_label)
        print('fake')
        print(out_d[0])

        out_r = dis(r_batch, pred_label)
        print('real')
        print(out_r[0])
        # for check output
        # add "return image, target, self.paths[idx]" to __getitem__ of ClassImageLoader
        # path = data[2]
        # for _ in path:
        #     shutil.copy(_, os.path.join(args.output_dir, 'out'))
        # # _ = [shutil.copy(_, os.path.join(args.output_dir, 'out')) for _ in path]
        # [save_image(output, os.path.join(args.output_dir, 'out',
        #             '{}_'.format(path[j].split('/')[-1].split('.')[0])
        #                          + s_li[r_cls[j]] + '.png'),
        #             normalize=True)
        #     for j, output in enumerate(out_t)]
 
        # # [save_image(out[(r_cls == j)], os.path.join(args.output_dir, 'out', s_li[j]+'_{}.png'.format(i)), normalize=True) for j in range(5) if len((r_cls == j).nonzero()) != 0]
        # # if i>20: exit()

        c_preds = torch.argmax(classifer(out_t).detach(), 1)
        cls_li.append(torch.cat([r_cls.int().cpu().view(r_cls.size(0), -1),
                                c_preds.int().cpu().view(c_preds.size(0), -1)], 1))
    all_res = torch.cat(cls_li, 0).numpy()
    y_true, y_pred = (all_res[:, 0], all_res[:, 1])

    table = classification_report(y_true, y_pred)

    print(table)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(s_li)))
    df = pd.DataFrame(data=matrix, index=s_li, columns=s_li)
    df.to_pickle(os.path.join(args.output_dir, 'cm.pkl'))

    plot = sns.heatmap(df, square=True, annot=True, fmt='d')
    fig = plot.get_figure()
    fig.savefig(os.path.join(args.output_dir, 'pr_table.png'))
