import transformers
from torch import nn
from tcn import TemporalConvNet


'''
TransTCN2: Technically 1st model we came up with but failed
Realized this architecture will not work and will discuss in project in further detail
Here in case need analysis
'''
class TransTCN2(nn.Module):
    def __init__(self, classes, input_size, num_channels, kernel_size=2, dropout=0.3, n=3, hidden_state=768):
        super(TransTCN2, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.drop = nn.Dropout(p=dropout)
        self.finalLinear = nn.Linear(hidden_state, classes)
        self.linear = nn.Linear(hidden_state, 150)
        self.sm = nn.Softmax(dim=1)
        self.n=n
        self.relu = nn.ReLU()

    #(bert -> tcn) * n -> bert -> linear -> softmax
    def forward(self, input_ids, attention_mask):
        output = input_ids
        for i in range(self.n):
            output = self.bert_model(output, attention_mask)
            output = output[1]
            output = output.unsqueeze(dim=1)
           
            output = self.tcn(output).squeeze(dim=1)
            
            output = self.linear(output)
            output = self.relu(output)
            output = output.long()

        output = self.bert_model(output, attention_mask)
        output = output[1]
        result = self.drop(output)
        result = self.finalLinear(output)

        return result
