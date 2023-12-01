from transformers import BertModel,BertTokenizer  #导入Bert模型和分词器


#分类任务时获得模型参数量代码
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num} #total_num:102267648 trainable_num:102267648


#模型的位置 Bert模型的参数和设置
bert_path = "bert-base-chinese"
#从文件夹下加载模型
# 根据config来创建模型，给到外面文件夹便会自动找到config文件
#查看bert的代码：BertModel上右击，Go to->declearation or usrages
model = BertModel.from_pretrained(bert_path)
# config.json中：设置"hidden_size": 768, bert中一个token的维度是768
# "intermediate_size": 3072,从768到3072 bert流程图中feed forward的linear变化
# "num_attention_heads": 12,多头注意力机制，在12个头中分别计算q，k，v
# "vocab_size": 21128 词表，有多少个字
print(get_parameter_number(model)) #查看bert模型有多少参数 {'Total': 102267648, 'Trainable': 102267648}有一亿多个

###bert模型架构的参数计算
#embadding层 token:21128 +句子编码 +位置编码每个位置是一个768维的向量
emb_para=21128*768+2*768+512*768
#自注意力层 方阵q,k,v + Add&form +feed forward:768*3072+3027*768
self_att_para=768*768*3+768*768+768*3072+3027*768
#总参数 pooler output有时候有参数有时候没参数，图像的pooler就没参数
para=emb_para+self_att_para*12
print(para) #101140992 也是一亿多但不相等，因为没加bias

#如何分词
#自动读tokenizer_config.json
mytokenizer=BertTokenizer.from_pretrained(bert_path)

input="早上好"
#adding="max_length":当长度不够时要padding,以max_length的方式
#需要张量形式的话,return_tensors="pt"可以把数据转成onn格式，可以读数据读得很快，并转成张量形式
out=mytokenizer(input,return_tensors="pt",truncation=True,padding="max_length",max_length=128)
print(out)
#修改前：
#{'input_ids': [101, 2769, 4263, 872, 102], 三个字但输出长度为5，因为一句话开头加了token:[CLS]是101,结尾加了token:[SEP]是102
# 'token_type_ids': [0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1]}
#修改后
#{'input_ids': [101, 2769, 4263, 872, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
#mask前五个是1，后面全是0