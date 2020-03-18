import argparse
import pickle
import os

import pandas as pd
import numpy as np
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str,
                    default='/mnt/8THDD/takamuro/dataset/photos_usa_2016')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/okada/from_nitta/parm_0.3/sepalated_data.pkl')
parser.add_argument('--save_path', type=str, default='cp/estimator/single')
parser.add_argument('--name', type=str, default='noname-estimator')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--mode', type=str, default='T', help='T(Train data) or E(Evaluate data)')
parser.add_argument('--multi', action='store_true')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from dataset import FlickrDataLoader
from sampler import ImbalancedDatasetSampler
from ops import soft_transform, l1_loss



comment = '_lr-{}_bs-{}_ne-{}_x{}_name-{}'.format(args.lr,
                                                  args.batch_size,
                                                  args.num_epoch,
                                                  args.input_size,
                                                  args.name)
writer = SummaryWriter(comment=comment)

save_dir = os.path.join(args.save_path, args.name)
os.makedirs(save_dir, exist_ok=True)

# load data
df = pd.read_pickle(args.pkl_path)
print('{} data were loaded'.format(len(df)))

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(args.input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
            brightness=0.5,
            contrast=0.3,
            saturation=0.3,
            hue=0
        ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((args.input_size,)*2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform = {'train': train_transform, 'test': test_transform}

# train_data_rate = 0.5
# pivot = int(len(df) * train_data_rate)

# if args.mode == 'P':
#     df_sep = {'train': df[:pivot], 'test': df[pivot:]}
# elif args.mode == 'E':  # for evaluation
#     df_sep = {'train': df[pivot:], 'test': df[:pivot]}
# else:
#     raise NotImplementedError

if args.mode == 'T':
    df_sep = {'train': df[df['mode'] == 'train'],
              'test': df[df['mode'] == 'test']}
elif args.mode == 'E':  # for evaluation
    df_sep = {'train': df[df['mode'] == 'val'],
              'test': df[df['mode'] == 'test']}
else:
    raise NotImplementedError

del df
cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed', 'rain']
loader = lambda s: FlickrDataLoader(args.image_root, df_sep[s],
                                    cols, transform[s])

train_set = loader('train')
test_set = loader('test')

train_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=ImbalancedDatasetSampler(train_set),
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

test_loader = torch.utils.data.DataLoader(
        test_set,
        sampler=ImbalancedDatasetSampler(test_set),
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

num_classes = train_set.num_classes

model = models.resnet101(pretrained=False, num_classes=num_classes)
model.cuda()
if args.multi:
    model = nn.DataParallel(model)

# train setting
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

criterion = nn.MSELoss()

eval_per_iter = 100
save_per_epoch = 5
global_step = 0

tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
for epoch in tqdm_iter:
    loss_li = []
    mse_li = []
    diff_li = []
    for i, data in enumerate(train_loader, start=0):
        inputs, labels = (d.to('cuda') for d in data)
        # soft_labels = soft_transform(labels, std=0.1)

        tqdm_iter.set_description('Training [ {} step ]'.format(global_step))

        # optimize
        opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        diff = l1_loss(outputs.detach(), labels)
        mse_li.append(loss.item())
        diff_li.append(diff.item())

        if global_step % eval_per_iter == 0:
            mse_li_ = []
            diff_li_ = []
            for j, data_ in enumerate(test_loader, start=0):
                with torch.no_grad():
                    inputs_, labels_ = (d.to('cuda') for d in data_)
                    outputs_ = model(inputs_).detach()
                    mse_ = criterion(outputs_, labels_)
                    diff_ = l1_loss(outputs_, labels_)
                    mse_li_.append(mse_.item())
                    diff_li_.append(diff_.item())

            # write summary
            train_mse = np.mean(mse_li)
            train_diff = np.mean(diff_li)
            test_mse = np.mean(mse_li_)
            test_diff = np.mean(diff_li_)
            writer.add_scalars('mse_loss', {'train': train_mse,
                                            'test': test_mse}, global_step)
            writer.add_scalars('l1_loss', {'train': train_diff,
                                           'test': test_diff}, global_step)
            mse_li = []
            diff_li = []

        global_step += 1

    if epoch % save_per_epoch == 0:
        out_path = os.path.join(save_dir, 'est_resnet101_'+str(epoch)+'_step'+str(global_step)+'.pt')
        if args.multi:
            torch.save(model.module.state_dict(), out_path)
        else:
            torch.save(model, out_path)

print('Done: training')
