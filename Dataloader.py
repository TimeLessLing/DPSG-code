import numpy as np
import torch
from torch.autograd import Variable
from Dependency import readDepTree

# copy from https://github.com/zhangmeishan/BiaffineDParser

def read_corpus(file_path, vocab=None):
    data = []
    with open(file_path, 'r', encoding='UTF-8') as infile:
        for sentence in readDepTree(infile, vocab):
            data.append(sentence)
    return data

