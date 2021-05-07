import transformers
import torch

'''
Processes data so that it can be fed into our model. Basically tokenizes the data since that is what we trained our model on.
'''
'''
GLOBAL TOKENIZER: Will be using bert uncased
'''
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

'''
Used for training to process large amounts of data
WARNING: Labels returned here are from 0-4. Real labels are from 1-5. Make sure to +1 whatever prediction the model makes
'''
class ProcessData(torch.utils.data.Dataset):
    def __init__(self, data, max_len, labels):
        self.data = data
        self.max_len = max_len
        self.labels = labels
    
    def __getitem__(self, index):
        review = self.data[index]
        label = self.labels[index]
        return TOKENIZER.encode_plus(review, max_length=self.max_len, padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True), review, torch.tensor([label-1]).to(torch.long)
    
    def __len__(self):
        review_length = len(self.data)
        return review_length

'''
Takes in a text and tokenizes it with the tokenizer

text: the review itself (MAKE SURE IT IS CLEANED AKA .lower is called on the string)
maxLength: The max length for each review. If it is shorter it is padded else it is clipped.

Returns a dictionary containing 3 things: 'input_ids', 'token_type_ids' and 'attention_mask'
We will be using the 'input_ids' and 'attention_masks' for our model
'''
def tokenize(text, maxLength):
    text = text.lower()
    return TOKENIZER.encode_plus(text, max_length=maxLength, padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True)