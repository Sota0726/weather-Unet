import argparse
# import pickle
import os

import pandas as pd
import numpy as np
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str,
                    default='/mnt/fs2/2019/takamuro/db/photos_usa_2016/')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/okada/'
                            'b4_sys/search_parm_new2/parm_0.3/'
                            'sepalated_data.pkl')
parser.add_argument('--name', type=str, default='noname_classifer')
parser.add_argument('--save_path', type=str, default='cp/classifier')
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='P')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
# import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from dataset import FlickrDataLoader
from sampler import ImbalancedDatasetSampler

os.makedirs(args.save_path, exist_ok=True)


def precision(outputs, labels):
    out = torch.argmax(outputs, dim=1)
    return torch.eq(out, labels).float().mean()


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

if args.mode == 'P':
    df_sep = {'train': df[df['mode'] == 'train'],
              'test': df[df['mode'] == 'test']}
elif args.mode == 'E':  # for evaluation
    df_sep = {'train': df[df['mode'] == 'test'],
              'test': df[df['mode'] == 'train']}
else:
    raise NotImplementedError

del df

cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed', 'rain']
loader = lambda s: FlickrDataLoader(args.image_root, df_sep[s],
                                    cols, transform[s], class_id=True)

train_set = loader('train')
test_set = loader('test')

train_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=ImbalancedDatasetSampler(train_set),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=4)

test_loader = torch.utils.data.DataLoader(
        test_set,
        sampler=ImbalancedDatasetSampler(test_set),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=4)

num_classes = len(train_set.cls_li)

# modify exist resnet101 model
model = models.resnet101(pretrained=False, num_classes=num_classes)
model.cuda()

# train setting
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
global_step = 0
eval_per_iter = 500
display_per_iter = 100
save_per_epoch = 5

tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)

comment = '_lr-{}_bs-{}_ne-{}_x{}_name-{}'.format(args.lr,
                                                  args.batch_size,
                                                  args.num_epoch,
                                                  args.input_size,
                                                  args.name)
writer = SummaryWriter(comment=comment)

loss_li = []
prec_li = []

for epoch in tqdm_iter:

    for i, data in enumerate(train_loader, start=0):
        inputs, w_info, labels = (d.to('cuda') for d in data)
        opt.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        prec = precision(outputs, labels)
        loss_li.append(loss.item())
        prec_li.append(prec.item())

        loss.backward()
        opt.step()

        if global_step % eval_per_iter == 0:

            loss_li_ = []
            prec_li_ = []
            for j, data_ in enumerate(test_loader):
                with torch.no_grad():
                    inputs_, w_info_, labels_ = (d.to('cuda') for d in data_)
                    predicted = model(inputs_)
                    loss_ = criterion(predicted, labels_)
                    prec_ = precision(predicted, labels_)
                    loss_li_.append(loss_.item())
                    prec_li_.append(prec_.item())

            writer.add_scalars('loss', {
                'train': np.mean(loss_li),
                'test': np.mean(loss_li_)
                }, global_step)
            writer.add_scalars('precision', {
                'train': np.mean(prec_li),
                'test': np.mean(prec_li_)
                }, global_step)
            loss_li = []
            prec_li = []

        global_step += 1

    if epoch % save_per_epoch == 0:
        tqdm_iter.set_description('{} iter: Training loss={:.5f} \
            precision={:.5f}'.format(
                                    global_step,
                                    np.mean(loss_li),
                                    np.mean(prec_li)
            ))

        out_path = os.path.join(args.save_path, 'resnet101_epoch'+str(epoch)+'_step'+str(global_step)+'.pt')
        torch.save(model, out_path)

print('Done: training')
