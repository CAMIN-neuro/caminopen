import os
from scipy import io
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from os.path import join
import torch
from torch import nn
import ast
from ast import literal_eval

# load NYU & TCD
path='/camin1/yrjang/Autism classification/abide2/diff/Schaefer_200/'
file_list=os.listdir(path)
file_list_py=[file for file in file_list if file.endswith('_DC.mat')]
conZ_1r=[]
for i in file_list_py:
    mat_file=io.loadmat(path+i)
    connmat=mat_file['diff_ConnMat']
    conZ_1r.append(connmat)
conZ_1r=np.array(conZ_1r)

# load sdsu
sdsu = np.load('/home/yrjang/Autism classification/sdsu_file/sdsu_sym_matrix.npy')
sdsu_age = np.load('/home/yrjang/Autism classification/sdsu_file/sdsu_age.npy')
sdsu_dx = np.load('/home/yrjang/Autism classification/sdsu_file/sdsu_dx.npy')
sdsu_sex = np.load('/home/yrjang/Autism classification/sdsu_file/sdsu_sex.npy')
sdsu_site = np.full((57,),3)



file=pd.read_excel('/camin1/yrjang/Autism classification/abide2/subjects.xlsx')

# Age & Site & Sex control
grp = file['Gk_II'].values
age = file['Ak_II'].values
sex = file['GENDER_II'].values
site = file['Sk_II'].values

grp_ = np.where(grp=='ASD', 1, grp)
grp_ = np.where(grp_=='CONTROL', 2, grp_)
grp = grp_.astype('int64')

grp = np.concatenate((grp,sdsu_dx))
age = np.concatenate((age,sdsu_age))
conZ_1r = np.concatenate((conZ_1r,sdsu),axis=0)


sex_ = np.where(sex=='M', 1, sex)
sex_ = np.where(sex_=='F', 2, sex_)
sex = sex_.astype('int64')

sex = np.concatenate((sex, sdsu_sex))

site_ = np.where(site=='TCD', 1, site)
site_ = np.where(site_=='NYU', 2, site_)
site = site_.astype('int64')

site = np.concatenate((site,sdsu_site))

NumSubj = np.size(grp)

NumROI=200
ConnMatZ2 = np.zeros((NumSubj, NumROI,NumROI))
ConnMatZ2_reg = np.zeros((NumSubj, NumROI,NumROI))

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
for num in range(0,len(feat2)):
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
df_con1=[x[i] for i in range(0,len(feat2))]

sdsu_dx_str = []
for i in sdsu_dx:
    if i == 1:
        sdsu_dx_str.append('ASD')
    else:
        sdsu_dx_str.append('CONTROL')

df_sdsu = pd.DataFrame({'Gk_II': sdsu_dx_str, 'Ak_II': sdsu_age, 'GENDER_II': sdsu_sex})
file = pd.concat([file,df_sdsu],ignore_index=True)

file['data']=df_con1



# Autoencoder code
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class AutoEncoder(nn.Module):
    def __init__(self, dropout, layer, unit_):
        super(AutoEncoder, self).__init__()
        self.dropout = nn.Dropout(p= dropout) # original 0.3
        self.layer = layer

        # set the number of unit
        self.fea_orig = 9900  # [630, 420, 200, 120]
        self.fea1 = 7700
        self.fea2 = 5500
        self.fea3 = 2930
        self.fea4 = 900
        self.fea5 = 200

        if layer == 'add':
            self.fea_orig = 9900  # [630, 420, 200, 120]
            self.fea1 = 7700
            self.fea2 = 5500
            self.fea3 = 4400
            self.fea4 = 2930
            self.fea5 = 900
            self.fea6 = 200

        if unit_[0] == 7700:
            print('origin')

        if unit_[0] == 7000:
            self.fea1 = 7000
            self.fea2 = 5000
            self.fea3 = 2000
            self.fea4 = 1000

            if self.layer == 'delete':
                self.fea4 = 200

        if unit_[0] == 8000:
            self.fea1 = 8000
            self.fea2 = 5000
            self.fea3 = 1000
            self.fea4 = 500

            if self.layer == 'delete':
                self.fea4 = 200


        self.encoder1 = nn.Sequential(
            nn.Linear(self.fea_orig, self.fea1),
            nn.Tanh())

        self.encoder2 = nn.Sequential(
            nn.Linear(self.fea1, self.fea2),
            nn.Tanh())

        self.encoder3 = nn.Sequential(
            nn.Linear(self.fea2, self.fea3),
            nn.Tanh())

        self.encoder4 = nn.Sequential(
            nn.Linear(self.fea3, self.fea4),
            nn.Tanh())

        self.decoder4 = nn.Sequential(
            nn.Linear(self.fea4, self.fea3),
            nn.Tanh())

        self.decoder3 = nn.Sequential(
            nn.Linear(self.fea3, self.fea2),
            nn.Tanh())

        self.decoder2 = nn.Sequential(
            nn.Linear(self.fea2, self.fea1),
            nn.Tanh())

        self.decoder1 = nn.Sequential(
            nn.Linear(self.fea1, self.fea_orig))

        if self.layer == 'origin':
            self.encoder5 = nn.Sequential(
                nn.Linear(self.fea4, self.fea5),
                nn.Tanh())

            self.decoder5 = nn.Sequential(
                nn.Linear(self.fea5, self.fea4),
                nn.Tanh())

        if self.layer == 'add':
            self.encoder5 = nn.Sequential(
                nn.Linear(self.fea4, self.fea5),
                nn.Tanh())

            self.decoder5 = nn.Sequential(
                nn.Linear(self.fea5, self.fea4),
                nn.Tanh())

            self.encoder6 = nn.Sequential(
                nn.Linear(self.fea5, self.fea6),
                nn.Tanh())

            self.decoder6 = nn.Sequential(
                nn.Linear(self.fea6, self.fea5),
                nn.Tanh())



    #             nn.Tanh())

    def forward(self, x):
        if self.layer == 'origin':
            x = self.dropout(x)
            e1 = self.encoder1(x)
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)
            latent = self.encoder5(e4)
            d5 = self.decoder5(latent)
            d4 = self.decoder4(d5)
            d3 = self.decoder3(d4)
            d2 = self.decoder2(d3)
            x = self.decoder1(d2)

        if self.layer == 'add':
            x = self.dropout(x)
            e1 = self.encoder1(x)
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)
            e5 = self.encoder5(e4)
            latent = self.encoder6(e5)
            d6 = self.decoder6(latent)
            d5 = self.decoder5(d6)
            d4 = self.decoder4(d5)
            d3 = self.decoder3(d4)
            d2 = self.decoder2(d3)
            x = self.decoder1(d2)

        if self.layer == 'delete':
            x = self.dropout(x)
            e1 = self.encoder1(x)
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            latent = self.encoder4(e3)
            d4 = self.decoder4(latent)
            d3 = self.decoder3(d4)
            d2 = self.decoder2(d3)
            x = self.decoder1(d2)

        return x, latent

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


###########################################################################################33
from copy import deepcopy
from torch.optim import Adam, lr_scheduler

dropout_list = [0.3, 0.1, 0.4]
lr_list = [0.00008, 0.0001, 0.001]
layer_list = ['origin','add','delete']
unit_list = [[7700,5500,2930,900],[7000,5000,2000,1000],[8000, 5000, 1000,500]]


#ASD / Control data split
file_asd= file[file['Gk_II']=='ASD']
file_con= file[file['Gk_II']=='CONTROL']

train1_asdf, val_asdf = train_test_split(file_asd, train_size=0.75, shuffle=True, random_state=3)
train_asdf, test_asdf = train_test_split(train1_asdf, train_size=0.75, shuffle=True, random_state=3)

train_asd_d_=[]
train_asd_d=train_asdf['data'].values
val_asd_d=val_asdf['data'].values
test_asd_d=test_asdf['data'].values

train1_conf, val_conf = train_test_split(file_con, train_size=0.75, shuffle=True, random_state=3)
train_conf, test_conf = train_test_split(train1_conf, train_size=0.75, shuffle=True, random_state=3)

train_con_d=train_conf['data'].values
val_con_d=val_conf['data'].values
test_con_d=test_conf['data'].values

# Data_loader
batch_size=2
asd_train_loader = DataLoader(train_asd_d, batch_size=batch_size, shuffle=True, drop_last=True)
asd_val_loader = DataLoader(val_asd_d, batch_size=batch_size, shuffle=True, drop_last=True)
asd_test_loader = DataLoader(test_asd_d, batch_size=batch_size, shuffle=True, drop_last=True)

con_train_loader = DataLoader(train_con_d, batch_size=batch_size, shuffle=True, drop_last=True)
con_val_loader = DataLoader(val_con_d, batch_size=batch_size, shuffle=True, drop_last=True)
con_test_loader = DataLoader(test_con_d, batch_size=batch_size, shuffle=True, drop_last=True)



for what in ['dropout','lr','layer','unit']:
    dropout = dropout_list[0]
    lr = lr_list[0]
    nlayer = layer_list[0]
    unit = unit_list[0]

    for grid in [1,2]:

        if what =='dropout':
            dropout = dropout_list[grid]

        if what =='lr':
            lr = lr_list[grid]

        if what =='layer':
            nlayer = layer_list[grid]

        if what =='unit':
            unit = unit_list[grid]

        print("\n")
        print("Start ---")
        print('model_d{}_u{}_layer{}_lr{} \n\n'.format(dropout, unit[0], nlayer, lr))


        layer = 'AE'

        model = AutoEncoder(layer=nlayer, unit_=unit, dropout=dropout).to(device='cuda:0')


        #lr = 0.0001
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        main_path = '/camin1/yrjang/Autism classification/Auto_revision/SDSU/grid_sdsu/control/model_d{}_u{}_layer{}_lr{}'.format(dropout,unit[0],nlayer, lr)
        #save_path = join(main_path, 'log')

        optimizer = torch.optim.ASGD(model.parameters(), lr=lr, weight_decay=0.1)
        lr_decay = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        criterion = nn.MSELoss(reduction='sum')

        file_list = os.listdir(main_path)
        load_test_model = join(main_path,file_list[0])
        model.load_state_dict(torch.load(load_test_model),strict=False)



        test_epoch_loss, x_test, x_hat_test, recon_loss_list_test, latent_test = fit(model, con_test_loader, layer, loss_show=True)



        x_test_n = []
        x_hat_test_n = []
        #x_n=[]
        #x_hat_n=[]
        for i in range(0,3):
            x_n=[]
            x_hat_n = []
            for num in range(0, 9900):
                x_n.append(x_test[i][num].cpu().detach().numpy().tolist())
                x_hat_n.append(x_hat_test[i][num].cpu().detach().numpy().tolist())
            x_test_n.append(x_n)
            x_hat_test_n.append(x_hat_n)


        import scipy.stats as stats
        from sklearn.metrics import mean_absolute_error, r2_score

        mean_cor = []
        mean_pvalue = []

        for num in range(0, np.array(x_test_n).shape[0]):
            cor, p = stats.pearsonr(x_test_n[num], x_hat_test_n[num])
            mean_cor.append(cor)
            mean_pvalue.append(p)
        cor_m = np.mean(mean_cor)
        p_m = np.mean(mean_pvalue)


        print("-------------------------------------")
        print("correlation coefficient : ", cor_m, "p-value : ", p_m)
        print("r2_score : ", r2_score(x_test_n,x_hat_test_n))
        print("MAE : ", mean_absolute_error(x_test_n, x_hat_test_n))

        # save result
        file = open('/home/yrjang/Autism classification/Auto_revision/grid_search/control/model_d{}_u{}_layer{}_lr{}_control_test.txt'.format(dropout, unit[0],
                                                                                                  nlayer, lr), 'w')
        file.write("correlation coefficient_:{}       p-value_:{}\n".format(cor_m, p_m))
        file.write("r2_score:{}       MAE_:{}\n".format(r2_score(x_test_n,x_hat_test_n),
                                                        mean_absolute_error(x_test_n, x_hat_test_n)))
        file.close()

        print('Finish : model_d{}_u{}_layer{}_lr{} \n\n'.format(dropout, unit[0], nlayer, lr))
        print('')
print('')

'''
y=x
plt.figure(figsize=(17,17))
plt.scatter(x_test_n[0],x_hat_test_n[0])
plt.plot(x,y)
plt.xlabel('Actual',fontsize=20)
plt.ylabel('Predicted',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.axis([-20000,20000,-20000,20000])
plt.show()


plt.figure(figsize=(17,17))
plt.scatter(x_test_n[0],x_hat_test_n[0],color='gray',s=200)
z=np.polyfit(x_test_n[0],x_hat_test_n[0],1)
p=np.poly1d(z)
plt.plot(x,p(x),color='gray')
plt.axis([-20000,20000,-20000,20000])
plt.savefig('con.png')
np.save('con_scatter_x_test.npy',x_test_n[0])
np.save('con_scatter_x_test_hat.npy',x_hat_test_n[0])

'''
"""
def fit(model, dataloader, layer, loss_show=True):
    model.train()
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

        optimizer.zero_grad()

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

        running_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()

    #         print('output',output[:,:3])
    #         print('data',data[:,:3])

    test_loss = running_loss / len(dataloader.dataset)
    if loss_show:
        print(f"Train Loss: {test_loss:.4f}")


    return test_loss, x_list, x_hat_list,  recon_loss_list,  latent_test_list"""
