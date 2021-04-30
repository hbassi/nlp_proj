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
device = 'cpu' if torch.cuda.is_available() else 'cpu'
#print(device)
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
        self.finalLinear = nn.Linear(hidden_state, classes)
        self.linear = nn.Linear(hidden_state,150)
        self.sm = nn.Softmax(dim=1)
        self.n=n
        self.relu = nn.ReLU()

    #(bert -> tcn) * n -> bert -> linear -> softmax
    def forward(self, input_ids, attention_mask):
        #print(input_ids,attention_mask )
        output = input_ids
        for i in range(self.n):
            #print(i)
            output = self.bert_model(output, attention_mask)
            output = output[1]
            #print('BERT Output: ', output)
            output = output.unsqueeze(dim=1)
            #print('Unsqueezed Output: ', output)
            #import pdb; pdb.set_trace()
           
            output = self.tcn(output).squeeze(dim=1)
            #import pdb; pdb.set_trace()
            

            output = self.linear(output)
            output = self.relu(output)
            output = output.long()
            #print(output.shape)



        output = self.bert_model(output, attention_mask)
        output = output[1]
        result = self.drop(output)
        result = self.finalLinear(output)

        return result
