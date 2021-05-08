import transformers
import torch
from torch import nn
from tcn import TemporalConvNet

'''
MAKE SURE DEVICE IS SET CORRECTLY IN EVERY FILE.
'''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
The TransTCN architecture
May not be close to as good as bert, but its honest work
'''
class TransTCN(nn.Module):
    '''
    classes: number of ratings (1-5)
    num_augmentations: number of data augmentations
    input_size: should be the same as num_augmentations.
    num_channels: list of number of in/out channels for each layer. The first layer takes in_channel input_size
                  For ex) [4, 4, 4, 4, 1] = 3 layers: input_size, 4 | 4, 4 | 4, 1 |
    kernel_size: size of the filter in TCN
    dropout: dropout probability for TCN
    hidden_state: default 768, was size of the output of pretrained bert
    is_single_bert: True or False value. If True, uses 1 bert model for all 
                    num_augmentations augementations. Else, uses num_augmentations bert 
                    models for num_augmentations augementations.
    '''
    def __init__(self, classes, num_augmentations, input_size, num_channels, kernel_size=2, dropout=0.3, hidden_state=768, is_single_bert=False):
        super(TransTCN, self).__init__()
        self.num_augmentations = num_augmentations
        self.is_single_bert = is_single_bert
        self.berts = []
        if self.is_single_bert:
            self.berts += [transformers.BertModel.from_pretrained('bert-base-uncased').to(DEVICE)]
        else:
            for i in range(num_augmentations):
                self.berts.append(transformers.BertModel.from_pretrained('bert-base-uncased').to(DEVICE))
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.finalLinear = nn.Linear(hidden_state, classes)
        
    def concatBerts(self, bert_outputs):
        concatBert = bert_outputs[0][1].unsqueeze(dim=1)
        for i in range(1, len(bert_outputs)):
            concatBert = torch.cat((concatBert, bert_outputs[i][1].unsqueeze(dim=1)), dim=1)
        return concatBert
    
    def forward(self, input_ids, attention_masks):
        bert_outputs = []
        for i in range(self.num_augmentations):
            if self.is_single_bert:
                bert_outputs.append(self.berts[0](input_ids[i], attention_masks[i]))
            else:
                bert_outputs.append(self.berts[i](input_ids[i], attention_masks[i]))
        output = self.concatBerts(bert_outputs)
        output = self.tcn(output).squeeze(dim=1)
        output = self.finalLinear(output)
        return output