import pandas as pd
from sklearn.model_selection import train_test_split
from data_processing import ProcessData
import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from architects.transtcn_model import TransTCN
from trainer_utils import trainingTransTCN, evaluateTransTCN
from tqdm import trange

'''
MAKE SURE DEVICE IS SET CORRECTLY IN EVERY FILE.
'''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
Hyperparameters
'''
TRAIN_TEST_SIZE = 0.1
VAL_TEST_SIZE = 0.4
RANDOM_STATE = 123

TEST_SIZE = 500
MAX_LENGTH = 150

BATCH_SIZE = 16
PARAMS = {'batch_size': BATCH_SIZE,
          'num_workers': 0}

LEARNING_RATE = 5e-5
NUM_CHANNELS = [4, 4, 1]
NUM_AUGMENTATIONS = 4
IS_SINGLE_BERT = True
NUM_EPOCHS = 3


'''
The training set
'''
data = pd.read_json('yelp_review_training_dataset.jsonl', lines=True)

'''
All of the augmentations on the training set
'''
delete_aug_data = pd.read_csv('augmentations/delete_aug.csv', header=None)
swap_aug_data = pd.read_csv('augmentations/swap_aug.csv', header=None)
typo_aug_data = pd.read_csv('augmentations/typo_aug.csv', header=None)
synonym_aug_data = pd.read_csv('augmentations/synonym_aug.csv', header=None)

delete_aug_data.columns = delete_aug_data.iloc[0]
delete_aug_data = delete_aug_data[1:]
delete_aug_data['stars'] = delete_aug_data['stars'].astype('int64')

swap_aug_data.columns = swap_aug_data.iloc[0]
swap_aug_data = swap_aug_data[1:]
swap_aug_data['stars'] = swap_aug_data['stars'].astype('int64')

typo_aug_data.columns = typo_aug_data.iloc[0]
typo_aug_data = typo_aug_data[1:]
typo_aug_data['stars'] = typo_aug_data['stars'].astype('int64')

synonym_aug_data.columns = synonym_aug_data.iloc[0]
synonym_aug_data = synonym_aug_data[1:]
synonym_aug_data['stars'] = synonym_aug_data['stars'].astype('int64')

'''
Train test split on data and its augmentations
'''
X = data['text'][:TEST_SIZE]
y = data['stars'][:TEST_SIZE]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SIZE, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=VAL_TEST_SIZE, random_state=RANDOM_STATE)

delete_X = delete_aug_data['text'][:TEST_SIZE]
delete_y = delete_aug_data['stars'][:TEST_SIZE]
delete_X_train, delete_X_test, delete_y_train, delete_y_test = train_test_split(delete_X, delete_y, test_size=TRAIN_TEST_SIZE, random_state=RANDOM_STATE)
delete_X_val, delete_X_test, delete_y_val, delete_y_test = train_test_split(delete_X_test, delete_y_test, test_size=VAL_TEST_SIZE, random_state=RANDOM_STATE)

swap_X = swap_aug_data['text'][:TEST_SIZE]
swap_y = swap_aug_data['stars'][:TEST_SIZE]
swap_X_train, swap_X_test, swap_y_train, swap_y_test = train_test_split(swap_X, swap_y, test_size=TRAIN_TEST_SIZE, random_state=RANDOM_STATE)
swap_X_val, swap_X_test, swap_y_val, swap_y_test = train_test_split(swap_X_test, swap_y_test, test_size=VAL_TEST_SIZE, random_state=RANDOM_STATE)

typo_X = typo_aug_data['text'][:TEST_SIZE]
typo_y = typo_aug_data['stars'][:TEST_SIZE]
typo_X_train, typo_X_test, typo_y_train, typo_y_test = train_test_split(typo_X, typo_y, test_size=TRAIN_TEST_SIZE, random_state=RANDOM_STATE)
typo_X_val, typo_X_test, typo_y_val, typo_y_test = train_test_split(typo_X_test, typo_y_test, test_size=VAL_TEST_SIZE, random_state=RANDOM_STATE)

synonym_X = synonym_aug_data['text'][:TEST_SIZE]
synonym_y = synonym_aug_data['stars'][:TEST_SIZE]
synonym_X_train, synonym_X_test, synonym_y_train, synonym_y_test = train_test_split(synonym_X, synonym_y, test_size=TRAIN_TEST_SIZE, random_state=RANDOM_STATE)
synonym_X_val, synonym_X_test, synonym_y_val, synonym_y_test = train_test_split(synonym_X_test, synonym_y_test, test_size=VAL_TEST_SIZE, random_state=RANDOM_STATE)

'''
Tokenize the data
'''

tokenized_training_data = ProcessData(X_train.to_numpy().tolist(), MAX_LENGTH, y_train.to_numpy())
tokenized_validation_data = ProcessData(X_val.to_numpy().tolist(), MAX_LENGTH, y_val.to_numpy())
tokenized_test_data = ProcessData(X_test.to_numpy().tolist(), MAX_LENGTH, y_test.to_numpy())

tokenized_training_data_typo = ProcessData(typo_X_train.to_numpy().tolist(), MAX_LENGTH, y_train.to_numpy())
tokenized_validation_data_typo = ProcessData(typo_X_val.to_numpy().tolist(), MAX_LENGTH, y_val.to_numpy())
tokenized_test_data_typo = ProcessData(typo_X_test.to_numpy().tolist(), MAX_LENGTH, y_test.to_numpy())

tokenized_training_data_swap = ProcessData(swap_X_train.to_numpy().tolist(), MAX_LENGTH, y_train.to_numpy())
tokenized_validation_data_swap = ProcessData(swap_X_val.to_numpy().tolist(), MAX_LENGTH, y_val.to_numpy())
tokenized_test_data_swap = ProcessData(swap_X_test.to_numpy().tolist(), MAX_LENGTH, y_test.to_numpy())

tokenized_training_data_delete = ProcessData(delete_X_train.to_numpy().tolist(), MAX_LENGTH, y_train.to_numpy())
tokenized_validation_data_delete = ProcessData(delete_X_val.to_numpy().tolist(), MAX_LENGTH, y_val.to_numpy())
tokenized_test_data_delete = ProcessData(delete_X_test.to_numpy().tolist(), MAX_LENGTH, y_test.to_numpy())

tokenized_training_data_synonym = ProcessData(synonym_X_train.to_numpy().tolist(), MAX_LENGTH, y_train.to_numpy())
tokenized_validation_data_synonym = ProcessData(synonym_X_val.to_numpy().tolist(), MAX_LENGTH, y_val.to_numpy())
tokenized_test_data_synonym = ProcessData(synonym_X_test.to_numpy().tolist(), MAX_LENGTH, y_test.to_numpy())

'''
Dataloaders
'''
loader_tokenized_training_data = torch.utils.data.DataLoader(tokenized_training_data, **PARAMS)
loader_tokenized_training_data_typo = torch.utils.data.DataLoader(tokenized_training_data_typo, **PARAMS)
loader_tokenized_training_data_swap = torch.utils.data.DataLoader(tokenized_training_data_swap, **PARAMS)
loader_tokenized_training_data_delete = torch.utils.data.DataLoader(tokenized_training_data_delete, **PARAMS)
loader_tokenized_training_data_synonym = torch.utils.data.DataLoader(tokenized_training_data_synonym, **PARAMS)
loaders = [loader_tokenized_training_data, loader_tokenized_training_data_typo, loader_tokenized_training_data_swap, loader_tokenized_training_data_delete, loader_tokenized_training_data_synonym]

loader_tokenized_validation_data = torch.utils.data.DataLoader(tokenized_validation_data, **PARAMS)
loader_tokenized_validation_data_typo = torch.utils.data.DataLoader(tokenized_validation_data_typo, **PARAMS)
loader_tokenized_validation_data_swap = torch.utils.data.DataLoader(tokenized_validation_data_swap, **PARAMS)
loader_tokenized_validation_data_delete = torch.utils.data.DataLoader(tokenized_validation_data_delete, **PARAMS)
loader_tokenized_validation_data_synonym = torch.utils.data.DataLoader(tokenized_validation_data_synonym, **PARAMS)
validation_loaders = [loader_tokenized_validation_data, loader_tokenized_validation_data_typo, loader_tokenized_validation_data_swap, loader_tokenized_validation_data_delete, loader_tokenized_validation_data_synonym]

'''
MODEL
'''
model = TransTCN(classes=5, num_augmentations=NUM_AUGMENTATIONS, input_size=NUM_AUGMENTATIONS,num_channels=NUM_CHANNELS, is_single_bert=IS_SINGLE_BERT)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(loader_tokenized_training_data) * NUM_EPOCHS)

'''
Training loop
'''
for epoch in trange(NUM_EPOCHS):
    print('Epoch: ' , str(epoch + 1))
    print('==================================')
    # If the batch is not batch_size length, the training loop wont run that batch
    eps = len(X_train) % BATCH_SIZE != 0
    training_accuracy, training_loss = trainingTransTCN(model, loaders, len(X_train), eps, criterion, optimizer, scheduler)
    validation_accuracy, validation_loss = evaluateTransTCN(model, validation_loaders, len(X_val), eps, criterion)
    
    print('Training accuracy: ', training_accuracy )
    print('Training loss: ', training_loss)
    print('Validation accuracy: ', validation_accuracy)
    print('Validation loss: ', validation_loss)