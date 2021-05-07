import transformers
from torch import nn

'''
The BertNet
'''
class BERTNet(nn.Module):
    def __init__(self, classes):
        super(BERTNet, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(768, classes)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids, attention_mask)
        output = output[1]
        output = self.drop(output)
        output = self.linear(output)
        return output