import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

save_path = os.path.join('temp-check_est', 'all-est')
df = pd.read_pickle('/mnt/fs2/2019/okada/from_nitta/parm_0.3/sepalated_data_wo-outlier.pkl')
l1 = pd.read_pickle('/home/sota/data/m2/weather-Unet/l1.pkl')
pred = pd.read_pickle('/home/sota/data/m2/weather-Unet/pred.pkl')

df = df[df['mode'] == 'train']
cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed']
# cols = ['clouds']

df_ = df.loc[:, cols].fillna(0)
df_mean = df_.mean()
df_std = df_.std()
# df = df_ori[df_ori['mode'] == 'test']
df.loc[:, cols] = (df_ - df_mean) / df_std

for i, col in enumerate(cols):

    gt = df[col].tolist()

    plt.figure()
    plt.hist(gt)
    plt.savefig(os.path.join(save_path, '{}_gt_hist.jpg'.format(col)))

    plt.figure()
    plt.hist(l1[i])
    plt.savefig(os.path.join(save_path, '{}_l1_hist.jpg'.format(col)))

    plt.figure()
    plt.hist(pred[i], bins=np.arange(min(pred[i]), max(pred[i]), 0.25))
    plt.savefig(os.path.join(save_path, '{}_pred_hist.jpg'.format(col)))
