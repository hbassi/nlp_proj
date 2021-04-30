import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import transformers
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
from torch import nn
import torch.nn.functional as F
#from tqdm import trange
from tqdm.notebook import tqdm, trange
import os
#from torch.optim import AdamW
from transformers import AdamW, get_linear_schedule_with_warmup
from tcn import TemporalConvNet


class TransTCN(nn.Module):
    def __init__(self, classes, input_size, num_channels, kernel_size=2, dropout=0.3, n=3, hidden_state=768):
        super(TransTCN, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_state, classes)
        self.sm = nn.Softmax(dim=1)
        self.n=n

    #(bert -> tcn) * n -> bert -> linear -> softmax
    def forward(self, input_ids, attention_mask):
        #print(input_ids,attention_mask )
        output = input_ids
        for i in range(self.n):
            output = self.bert_model(output, attention_mask)
            output = output[1]
            output = self.tcn(output)
        output = self.bert_model(output, attention_mask)
        result = self.drop(output)
        result = self.linear(output)

        return result
