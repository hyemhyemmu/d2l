# 新版的d2l没有d2l.load_data_time_machine, 以下人为定义一个

import os
import re
import random
import torch
from torch.utils import data
from d2l import torch as d2l  

# 定义时间机器数据集的下载链接和校验码
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines):
    """将每行文本拆分成单词"""
    return [line.split() for line in lines]

def seq_data_iter_sequential(corpus_indices, batch_size, num_steps):
    """顺序采样生成小批量子序列"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus_indices) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus_indices[offset:offset + num_tokens])
    Ys = torch.tensor(corpus_indices[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y

def load_data_time_machine(batch_size, num_steps):
    """加载并返回 time machine 数据集的迭代器"""
    lines = read_time_machine()
    corpus = tokenize(lines)
    all_text = [word for line in corpus for word in line]
    vocab = d2l.Vocab(all_text)
    corpus_indices = [vocab[word] for word in all_text]
    return seq_data_iter_sequential(corpus_indices, batch_size, num_steps), vocab
