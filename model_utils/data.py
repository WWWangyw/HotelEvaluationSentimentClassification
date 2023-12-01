#数据部分代码，数据集：jiudian.txt
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split  #作用：把数据分为测试集和验证集
from tqdm import tqdm

def read_txt(path):
    data=[]
    label=[]
    with open(path,"r", encoding="utf-8") as f: #以utf-8的编码方式来读
        for i,line in enumerate(f):
            #第一行没用，不读
            if i==0:
                continue #跳过本次循环直接开始下次循环
            #数据集太大，先只取两头的试试，中间的先不用
            if i>200 and i<7700:
                break
            line=line.strip("\n") #去掉换行符
            line=line.split(",",1)#用逗号把标签和文本分开，1表示只分一次，所以只会分最开头的一次，后面的不管
            label.append(line[0])
            data.append(line[1])


    return data,label

# read_txt("../jiudian.txt") #../读上一层文件夹的文件，在运行main.py报错，因为从main.py，上一层是Bert找不到txt

class JDDataset(Dataset):
    def __init__(self,data,label):
        self.X=data
        self.Y=[int(i) for i in label] #对label中的每一个i都转为int型
        #分类的标签必须转为长整型，才不会报错
        self.Y=torch.LongTensor(self.Y)

    def __getitem__(self, item):
        return self.X[item],self.Y[item]

    def __len__(self):
        return len(self.Y)

#强行分的训练集和验证集，所以要设定验证集比例多大
def get_data_loader(path,batchsize,val_size=0.2):
    #读出数据和label
    data,label=read_txt(path)
    #把直接传入的数据集分割开
    #shuffle=True 是否把数据集打乱
    #stratify=label 是否按label比例取数据。例如label中有1000个是0，有500个是1，取0.2比例则取200个0和100个1
    train_x,val_x,train_y,val_y=train_test_split(data,label,test_size=val_size,shuffle=True,stratify=label)

    #创建数据集
    train_set=JDDataset(train_x,train_y)
    #创建验证集
    val_set=JDDataset(val_x,val_y)

    train_loader=DataLoader(train_set,batchsize)
    val_loader=DataLoader(val_set,batchsize)

    return train_loader,val_loader

#创建函数成功，下一步创建模型

#让这一步只在此文件夹中运行
if __name__=="__main__":
    read_txt("../jiudian.txt") #../读上一层文件夹的文件