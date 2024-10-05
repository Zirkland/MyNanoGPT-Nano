import torch
import torch.nn as nn
import math
from transformers import GPT2Config, GPT2Model
from DataloaderNano import hyperparameters

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=512):
        super().__init__()
        self.pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        seq_length = x.size(1)
        pe = self.pe[:, :seq_length, :].to(x.device)
        return x + pe

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, 'embedding_dim必须是num_heads的整数倍'

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, embedding_dim = x.size()
        qkv = self.qkv(x).view(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)

        context = torch.matmul(attention, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, embedding_dim)
        return self.out(context)

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ffn_dim):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embedding_dim)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(hyperparameters['vocab_size'], hyperparameters['embedding_dim'])
        self.positional_encoding = PositionalEncoding(hyperparameters['embedding_dim'], hyperparameters['max_len'])
        self.multi_head_attention = MaskedMultiHeadAttention(hyperparameters['embedding_dim'], hyperparameters['num_heads'])
        self.layer_norm1 = LayerNorm(hyperparameters['embedding_dim'])
        self.feed_forward = FeedForward(hyperparameters['embedding_dim'], hyperparameters['ffn_dim'])
        self.layer_norm2 = LayerNorm(hyperparameters['embedding_dim'])
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hyperparameters['embedding_dim'], hyperparameters['vocab_size'])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.multi_head_attention(x, mask)
        x = self.layer_norm1(x)
        x = self.feed_forward(x)
        x = self.layer_norm2(x)
        x = self.dropout(x)
        logits = self.linear(x)
        return logits


# 加载GPT-2预训练参数
def load_gpt2_pretrained(model, model_name='gpt2'):
    gpt2_model = GPT2Model.from_pretrained(model_name)
    model_dict = model.state_dict()
    pretrained_dict = gpt2_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

# 使用示例

