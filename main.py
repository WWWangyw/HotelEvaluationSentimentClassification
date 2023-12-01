import torch.nn as nn
import torch
import random
import numpy as np
import os

from model_utils.data import get_data_loader
from model_utils.model import MyModel
from model_utils.train import train_val

#随机种子固定训练结果
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


model_name='MyModel'
num_class=2 #二分类任务，若做多分类任务可直接在这里改
batchSize=4
learning_rate=0.0001
loss=nn.CrossEntropyLoss()
epoch=5
device='cuda' if torch.cuda.is_available() else 'cpu'

data_path='jiudian.txt'
bert_path='bert-base-chinese'
save_path='model_save/'
seed_everything(1)

#读取数据集
train_loader, val_loader = get_data_loader(data_path, batchsize=batchSize)
model=MyModel(bert_path,device,num_class).to(device) #模型也要放在cpu上

#优化器
param_optimizer = list(model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.0001)
#学习率变化，让学习率一直在波动
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20,eta_min=1e-9)

trainpara = {'model': model,
             'train_loader': train_loader,
             'val_loader': val_loader,
             'scheduler': scheduler,
             'optimizer': optimizer,
             'learning_rate': learning_rate,
             'warmup_ratio' : 0.1,
             'weight_decay' : 0.0001,
             'use_lookahead' : True,
             'loss': loss,
             'epoch': epoch,
             'device': device,
             'save_path': save_path,
             'max_acc': 0.85,
             'val_epoch' : 1
             }
train_val(trainpara)