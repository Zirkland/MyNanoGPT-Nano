import glob
import json
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 指定模型名称
model_name = 'gpt2'

# 下载并加载GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("GPT-2 model and tokenizer have been successfully loaded.")

# 读取语料库
text = open('corpus.txt', 'r', encoding='utf-8').read()

# 定义一个函数将文本分割成较小的块
def chunk_text(text, chunk_size=10000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# 将文本分割成较小的块
chunks = chunk_text(text)

# 使用GPT-2开源代码中的tokenizer对每个块进行分词
data = []
max_length = 1024  # GPT-2's maximum sequence length
for chunk in chunks:
    tokens = tokenizer.encode(chunk, max_length=max_length, truncation=True)
    data.extend(tokens)

vocab = tokenizer.get_vocab()
print("Vocabulary size:", len(vocab))

# 定义编码与解码函数
encode = lambda s: [vocab[c] for c in s]
decode = lambda x: ''.join([vocab[i] for i in x])

# 得到训练数据与验证数据
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 超参字典，方便其他地方调用
hyperparameters = {
    'device': 'cuda',  # 启动我的4080laptop！！！
    'block_size': 128,  # 每次输入的长度
    'batch_size': 32,  # 每次输入的数量(一批多少个)
    'vocab_size': len(vocab),
    'embedding_dim': 768,  # 嵌入维度,在embedding与MaskedMultiHeadAttention中被调用
    'max_len': 512,  # 最大长度,在PositionalEncoding中被调用
    'num_heads': 12,  # 多头注意力的头数,在MaskedMultiHeadAttention中被调用
    'ffn_dim': 3072,  # 前馈神经网络的维度,在FeedForward中被调用
    'num_layers': 12,  # transformer块的数量
    'learning_rate': 1e-4,  # 学习率
    'max_iters': 100000,  # 最大迭代次数，就是总训练步数
    'eval_interval': 400,  # 每隔多少次迭代进行一次验证
    'log_interval': 250,  # 每隔多少次迭代输出一次日志
}

# 定义获取batch的函数，返回x,y，x是输入，y是输出，x和y的shape都是[batch_size,block_size]，y是x的后一个字符，这样就可以用x预测y，即语言模型，这里的x和y都是tensor
def get_batch(split):
    data = train_data if split == 'train' else val_data
    start_idx = torch.randint(0, len(data) - hyperparameters['block_size'], (hyperparameters['batch_size'],))  # 随机生成一组起始位置
    x = torch.stack([torch.tensor(data[i:i + hyperparameters['block_size']]) for i in start_idx])
    y = torch.stack([torch.tensor(data[i + 1:i + 1 + hyperparameters['block_size']]) for i in start_idx])
    return x.to(hyperparameters['device']), y.to(hyperparameters['device'])