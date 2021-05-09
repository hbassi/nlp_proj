from tqdm import tqdm
import torch
from torch import nn
import numpy as np
'''
MAKE SURE DEVICE IS SET CORRECTLY IN EVERY FILE.
'''
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def create_data_loader_iterators(data_loaders):
    result = []
    for data_loader in data_loaders:
        result += [iter(data_loader)]
    return result

def trainingTransTCN(model, data_loaders, size, eps, criterion, optimizer, scheduler):
    model = model.train()
    losses = []
    training_acc = 0
    iters = create_data_loader_iterators(data_loaders)
    for _ in tqdm(range(len(data_loaders[0]) - eps)):

        batch_ids_list = []
        batch_masks_list = []
        labels = None
        for i in iters:

            data = next(i)

            batch_ids = data[0]['input_ids']
            batch_ids = batch_ids.flatten().reshape((batch_ids.shape[0], batch_ids.shape[2]))
            batch_ids_list.append(batch_ids.to(DEVICE))

            batch_masks = data[0]['attention_mask']
            batch_masks = batch_masks.flatten().reshape((batch_masks.shape[0], batch_masks.shape[2]))
            batch_masks_list.append(batch_masks.to(DEVICE))
        
            labels = data[2]

        labels = labels.to(DEVICE)
        output = model(batch_ids_list, batch_masks_list)
  
        prediction = torch.max(output, 1)[1]

        training_loss = criterion(output, torch.flatten(labels))
        training_acc += torch.sum(prediction == torch.flatten(labels))

        losses.append(training_loss.item())
        training_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
      
    return training_acc / size, np.mean(losses)

def evaluateTransTCN(model, data_loaders, size, eps, criterion):
    model = model.eval()
    losses = []
    validation_acc = 0
    iters = create_data_loader_iterators(data_loaders)
    with torch.no_grad():
        for _ in tqdm(range(len(data_loaders[0]) - eps)):
            vbatch_ids_list = []
            vbatch_masks_list = []
            vlabels = None
            for i in iters:
                vdata = next(i)

                vbatch_ids = vdata[0]['input_ids']
                vbatch_ids = vbatch_ids.flatten().reshape((vbatch_ids.shape[0], vbatch_ids.shape[2]))
                vbatch_ids_list.append(vbatch_ids.to(DEVICE))

                vbatch_masks = vdata[0]['attention_mask']
                vbatch_masks = vbatch_masks.flatten().reshape((vbatch_masks.shape[0], vbatch_masks.shape[2]))
                vbatch_masks_list.append(vbatch_masks.to(DEVICE))

                vlabels = vdata[2]

            vlabels = vlabels.to(DEVICE)
            voutput = model(vbatch_ids_list, vbatch_masks_list)
            vprediction = torch.max(voutput, 1)[1]
                
            vloss = criterion(voutput, torch.flatten(vlabels))
            validation_acc += torch.sum(vprediction == torch.flatten(vlabels))
            losses.append(vloss.item())
    return validation_acc / size, np.mean(losses)