from architects.bertnet_model import BERTNet
from architects.transtcn_model import TransTCN
import torch

'''
Loads model for test_submission.py

Update this in the future if there is a better model that can be used
'''
def load_model(name):
    if name == 'bert':
        model = BERTNet(5)
        model.load_state_dict(torch.load('saved_models/epoch4_augmented_full_data_model.pth'))
        return model
    else:
        raise ValueError('model name: ' + name + ' is currently not supported')