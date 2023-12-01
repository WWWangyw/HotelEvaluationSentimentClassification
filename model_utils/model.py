#创建模型
#分类模型
import torch.nn as nn
import torch
from transformers import BertModel,BertTokenizer,BertConfig

#定义自己的模型
class MyModel(nn.Module):
    def __init__(self,bert_path,device,num_class):
        super(MyModel,self).__init__()
        self.device=device

        #定义一个bert模型
        #bert_path:下载的bert-base文件，自动读config文件知道架构，同时直接把模型参数加载好，从而创建Bert模型
        #from_pretrained():Bert可以用这个从文件夹读取设置
        self.bert=BertModel.from_pretrained(bert_path)
        self.tokenizer=BertTokenizer.from_pretrained(bert_path)
        self.cls_head=nn.Linear(768,2,num_class)



    def forward(self,text):
        #先用分词器处理这句话
        #得到字典，里面有三个'input_ids''token_type_ids'和'attention_mask'
        input= self.tokenizer(text,return_tensors="pt",truncation=True,padding="max_length",max_length=128)
        input_ids=input["input_ids"].to(self.device)
        token_type_ids=input["token_type_ids"].to(self.device)
        attention_mask=input["attention_mask"].to(self.device)

        #这些新产生的数据会在cpu上，但模型一般会放在gpu上，所以需要to(device)

        #把这些输入进模型
        #bert的输出有两个：sequence_out,pooler_out
        sequence_out,pooler_out= self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        out=self.cls_head(pooler_out)
        return out

#模型完成，下一步训练