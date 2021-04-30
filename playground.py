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

class ProcessData(torch.data.Dataset):
    def __init__(self, data, tokenizer, max_len, label):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label = label

    
    def __getitem__(self, index):
        review = self.data[index]
        return review
    
    def __len__(self):
        review_length = len(self.data)
        return review_length
    
    def data(self):
        tokenized = self.tokenizer.encode_plus(self.data, max_length=self.max_len, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')

# a = ProcessData(data, tokensizer, maxlen, label)
# tokenizedData = a.data()

# 1. Process data 
#  - obtain vocab, obtain translations (idx ->word & word -> index)
# 2. Created Bert Model (**********modify this in future for better models********)
# 3. Create ***Optimizer*** and epoch training loops (print accuracies)
# 4. Create loss fn
# 5. Get prelim results (creating output json)