import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
import logging
import os
from sklearn.model_selection import train_test_split
import collections
import six
import time


class Vocab():
    """
    根据本地的vocab文件，构造一个词表
    vocab = Vocab()
    print(vocab.itos)   # 得到一个列表，返回列表中的每一个词
    print(vocab.itos[2])   # 通过索引返回得到词表中对应的词
    print(vocab.stoi)   # 得到一个字典，返回词表中每个词的索引
    print(vocab.stoi['我']) # 通过单词返回得到词表中对应的索引
    print(len(vocab))   # 返回词表长度
    """
    UNK = "[UNK]"
    
    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))
    
    def __len__(self):
        return len(self.itos)

def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


def process_cache(unique_key=None):
    """
    数据预处理结果缓存修饰器
    :param : unique_key
    :return:
    """
    if unique_key is None:
        raise ValueError(
            "unique_key 不能为空, 请指定相关数据集构造类的成员变量，如['top_k', 'cut_words', 'max_sen_len']")

    def decorating_function(func):
        def wrapper(*args, **kwargs):
            logging.info(f" ## 索引预处理缓存文件的参数为：{unique_key}")
            obj = args[0]  # 获取类对象，因为data_process(self, file_path=None)中的第1个参数为self
            file_path = kwargs['file_path']
            file_dir = f"{os.sep}".join(file_path.split(os.sep)[:-1])
            file_name = "".join(file_path.split(os.sep)[-1].split('.')[:-1])
            paras = f"cache_{file_name}_"
            for k in unique_key:
                paras += f"{k}{obj.__dict__[k]}_"  # 遍历对象中的所有参数
            cache_path = os.path.join(file_dir, paras[:-1] + '.pt')
            start_time = time.time()
            if not os.path.exists(cache_path):
                logging.info(f"缓存文件 {cache_path} 不存在，重新处理并缓存！")
                data = func(*args, **kwargs)
                with open(cache_path, 'wb') as f:
                    torch.save(data, f)
            else:
                logging.info(f"缓存文件 {cache_path} 存在，直接载入缓存文件！")
                with open(cache_path, 'rb') as f:
                    data = torch.load(f)
            end_time = time.time()
            logging.info(f"数据预处理一共耗时{(end_time - start_time):.3f}s")
            return data

        return wrapper

    return decorating_function