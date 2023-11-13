import os
import skimage.io
import skimage.transform
from scipy import io
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from captum.attr import IntegratedGradients, LRP, NeuronIntegratedGradients
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from os.path import join
import torch
from torch import nn
import ast
from ast import literal_eval
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, r2_score


path='/camin1/yrjang/Autism classification/abide2/diff/Schaefer_200/'
file_list=os.listdir(path)
file_list_py=[file for file in file_list if file.endswith('_DC.mat')]
conZ_1r=[]
for i in file_list_py:
    mat_file=io.loadmat(path+i)
    connmat=mat_file['diff_ConnMat']
    conZ_1r.append(connmat)
conZ_1r=np.array(conZ_1r)

file=pd.read_excel('/camin1/yrjang/Autism classification/abide2/subjects.xlsx')

# Age & Site & Sex control
grp = file['Gk_II'].values
age = file['Ak_II'].values
sex = file['GENDER_II'].values
site = file['Sk_II'].values

grp_ = np.where(grp=='ASD', 1, grp)
grp_ = np.where(grp_=='CONTROL', 2, grp_)
grp = grp_.astype('int64')

sex_ = np.where(sex=='M', 1, sex)
sex_ = np.where(sex_=='F', 2, sex_)
sex = sex_.astype('int64')

site_ = np.where(site=='TCD', 1, site)
site_ = np.where(site_=='NYU', 2, site_)
site = site_.astype('int64')

NumSubj = np.size(grp)

NumROI=200
ConnMatZ2 = np.zeros((np.size(file_list_py), NumROI,NumROI))
ConnMatZ2_reg = np.zeros((np.size(file_list_py), NumROI,NumROI))

for nr1 in range(0,NumROI):
    for nr2 in range(0,NumROI):
        x = np.transpose([age, sex,site])
        y = np.expand_dims(conZ_1r[:,nr1,nr2], axis=1)
        lm = LinearRegression()
        lm.fit(x,y)
        ConnMatZ2_reg[:,nr1,nr2] = np.squeeze(y - lm.predict(x))

feat2=ConnMatZ2_reg

#Flatten
hemi=[]
hemi2=[]
for num in range(0,84):
    for Left in range(0,100):
        if i == 99:
            break
        for j in range (0+Left,99):
            hemi.append(feat2[num][Left][j+1])
    for right in range (100,200):
        if i == 199:
            break
        for r in range (100+(right-100),199):
            hemi.append(feat2[num][right][r+1])
    hemi2.append(hemi)
    hemi=[]
x=np.array(hemi2)
df_con1=[x[i] for i in range(0,84)]

file['data']=df_con1





# Autoencoder code
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

fea_orig = 9900   #[630, 420, 200, 120]
fea1 = 7700
fea2 = 5500
fea3 = 2930
fea4 = 900
fea5 = 200


# LeakyReLU ReLU Tanh
# Tanh: best performance

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.dropout = nn.Dropout(p=0.3 )
        self.encoder1 = nn.Sequential(
            nn.Linear(fea_orig, fea1),
            nn.Tanh())

        self.encoder2 = nn.Sequential(
            nn.Linear(fea1, fea2),
            nn.Tanh())

        self.encoder3 = nn.Sequential(
            nn.Linear(fea2, fea3),
            nn.Tanh())
        self.encoder4 = nn.Sequential(
            nn.Linear(fea3, fea4),
            nn.Tanh())
        self.encoder5 = nn.Sequential(
            nn.Linear(fea4, fea5),
            nn.Tanh())


    #             nn.Tanh())

    def forward(self, x):
        x = self.dropout(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        latent = self.encoder5(e4)

        return latent

def fit(model, dataloader, layer, loss_show=True):
    model.eval()
    running_loss = 0.0
    running_KLD_loss = 0.0

    y_list = []
    y_hat_list = []
    x_list = []
    x_hat_list = []
    recon_loss_list = []
    reg_loss_list = []
    vae_loss_list = []

    total_predict = []
    total_y_pred = []
    cla_loss_list = []
    latent_test_list = []

    alpha = 1
    for i, data in enumerate(dataloader):
        x = data[0]
        x = x.to(device)
        #x = x.view(x.size(0), -1)


        #         y = data[1]
        #         y = y.to(device)
        #         y = y.view(y.size(0), -1)
        #             print('y',y)
        if layer == 'AE':
            x_hat, latent_test = model(x.float())
            recon_loss = criterion(x.float(), x_hat.float())
            loss = recon_loss

            x_list.append(x)
            x_hat_list.append(x_hat)
            recon_loss_list.append(recon_loss.item())
            latent_test_list.append(latent_test)
        #         del y

        del x
        torch.cuda.empty_cache()


    #         print('output',output[:,:3])
    #         print('data',data[:,:3])

    test_loss = running_loss / len(dataloader.dataset)
    if loss_show:
        print(f"Train Loss: {test_loss:.4f}")


    return test_loss, x_list, x_hat_list,  recon_loss_list,  latent_test_list

from copy import deepcopy
from torch.optim import Adam, lr_scheduler

    #ASD / Control data split
file_asd= file[file['Gk_II']=='ASD']
file_con= file[file['Gk_II']=='CONTROL']
con_data = file_con['data'].values
a=file['data'].values
x_data=np.zeros((37,9900))
for i in range(0,37):
    for j in range(0,9900):
        x_data[i][j]=con_data[i][j]

def IG_grad(sig_idx):
    X = torch.tensor(x_data).float().to('cpu')  # [subj_num, Multi-gradients]
    X.requires_grad_()
    IG_grad_elem = []
    attr_list = []
    IG_sum = np.zeros(np.array(x_data).shape)
    for i in sig_idx:  # sig_idx
        #         attr, delta = lrp.attribute(X,target=0, return_convergence_delta=True)
        attr = nig.attribute(X, int(i))
        attr = attr.detach().numpy()
        IG_sum += attr
    return IG_sum.mean(axis=0), IG_sum

model = AutoEncoder().to('cpu')
main_path = "/camin1/yrjang/Autism classification/Checkpoint_auto/model"
save_path = join(main_path, 'log_c')
load_test_model = join(save_path,'_best_AE_epoch420.pkl')
model.load_state_dict(torch.load(load_test_model), strict=False)

# Model interpretation
nig = NeuronIntegratedGradients(model, model.encoder5)  # (model, target layer)
IG_testmean, IG_testsum = IG_grad(np.arange(200))

# (37,9900) IG save
np.save('con_zscore_9900_420.npy',IG_testsum)

#IG_vector_reconstruction
o_matrix = np.array(IG_testsum)
x_test_matrix = np.zeros((37,200, 200))
matrix_sub = np.zeros((37, 200, 200))
for tnum in range(0,37):
    for k in range(0, 100):
        if k == 99 or k == 199:
            x_test_matrix[tnum][k][k] = 0
        else:
            a = 4950 - ((99 - k) * (100 - k)) / 2
            a = int(a)
            change_matrix = o_matrix[tnum][a:a + (99 - k)]
            for n in range(0, 99):
                if n == 99 - (k):
                    break
                else:
                    x_test_matrix[tnum][k][k + n + 1] = change_matrix[n]
            for n in range(0, 99):
                if k == n:
                    x_test_matrix[tnum][k][k] = 0
    for k in range(100, 200):
        if k == 99 or k == 199:
            x_test_matrix[tnum][k][k] = 0
        else:
            a = 4950 * 2 - ((99 - (k - 100)) * (100 - (k - 100))) / 2
            a = int(a)
            change_matrix = o_matrix[tnum][a:a + (99 - (k - 100))]
            for n in range(0, 99):
                if n == 99 - (k - 100):
                    break
                else:
                    x_test_matrix[tnum][k][k + n + 1] = change_matrix[n]
            for n in range(100, 199):
                if k == n:
                    x_test_matrix[tnum][k][k] = 0
    matrix_sub[tnum] = x_test_matrix[tnum].T + x_test_matrix[tnum]

print(IG_testmean.shape)

new = np.zeros((1,37))
zsco_sub = np.zeros((37,200,200))
for col in range(0,200):
    for nu in range(0,200):
        for num in range(0,37):
            new[0][num] = matrix_sub[num][col][nu]
        score = zscore(new[0])
        for num1 in range(0,37):
            zsco_sub[num1][col][nu] = score[num1]
        new = np.zeros((1,37))

np.save('con_zscore_sub.npy',zsco_sub)







