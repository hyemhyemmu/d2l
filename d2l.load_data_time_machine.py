# 新版的d2l没有d2l.load_data_time_machine，以下人为定义一个

import os
import re
import random
import torch
from torch.utils import data

# 定义时间机器数据集的下载链接和校验码
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

# 读取时间机器数据集并进行数据清洗
def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# 分词函数：将每行文本拆分成单词
def tokenize(lines):
    """将每行文本拆分成单词"""
    return [line.split() for line in lines]

# 加载数据并生成数据迭代器
def load_data_time_machine(batch_size, num_steps):
    """加载并返回time machine数据集的迭代器"""
    # 1. 读取和清洗数据
    lines = read_time_machine()
    
    # 2. 分词
    corpus = tokenize(lines)
    
    # 3. 创建词汇表
    # 将所有单词汇总到一个列表中
    all_text = [word for line in corpus for word in line]
    
    # 创建词汇表：统计词频
    vocab = d2l.Vocab(all_text)
    
    # 4. 将文本转为整数索引
    corpus_indices = [vocab[word] for line in corpus for word in line]
    
    # 5. 创建数据集
    # 以num_steps为每一序列的长度，batch_size为每一批次的大小
    def data_iter(batch_size, num_steps, corpus_indices):
        """生成数据迭代器"""
        # 计算每个小批量的起始位置
        num_examples = (len(corpus_indices) - 1) // num_steps
        example_indices = list(range(0, num_examples * num_steps, num_steps))
        
        # 按照batch_size划分数据
        random.shuffle(example_indices)
        
        # 生成每批次的数据
        def _next_batch():
            batch = torch.zeros((batch_size, num_steps), dtype=torch.long)
            for i in range(batch_size):
                start_idx = example_indices[i]  # 获取起始位置
                batch[i] = torch.tensor(corpus_indices[start_idx:start_idx + num_steps])
            return batch
        return _next_batch

    # 返回迭代器，生成每批次数据
    return data_iter(batch_size, num_steps, corpus_indices), vocab
