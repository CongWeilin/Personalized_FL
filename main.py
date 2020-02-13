#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


# In[2]:


# In[3]:


# guarantee reproducible results
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

start_time = time.time()

# define paths
path_project = os.path.abspath('..')

args = args_parser()

print(args)
exp_details(args)

if args.gpu:
    torch.cuda.set_device(args.gpu)
device = 'cuda' if args.gpu else 'cpu'

# load dataset and user groups
train_dataset, test_dataset, user_groups = get_dataset(args)

# BUILD MODEL
if args.model == 'cnn':
    # Convolutional neural netork
    if args.dataset == 'mnist':
        global_model = CNNMnist(args=args)
    elif args.dataset == 'fmnist':
        global_model = CNNFashion_Mnist(args=args)
    elif args.dataset == 'cifar':
        global_model = CNNCifar(args=args)

elif args.model == 'mlp':
    # Multi-layer preceptron
    img_size = train_dataset[0][0].shape
    len_in = 1
    for x in img_size:
        len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=args.num_classes)
else:
    exit('Error: unrecognized model')


# In[4]:


# Set the model to train and send it to device.
global_model.to(device)
global_model.train()
print(global_model)

# copy weights
global_weights = global_model.state_dict()


# In[5]:


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
    
def train_val_test(dataset, idxs):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    # split indexes for train, validation, and test (80, 10, 10)
    idxs_train = idxs[:int(0.8*len(idxs))]
    idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
    idxs_test = idxs[int(0.9*len(idxs)):]
    # print(len(idxs_train), len(idxs_val), len(idxs_test))

    trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                             batch_size=args.local_bs, shuffle=True)
    validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                             batch_size=int(len(idxs_val)/10), shuffle=False)
    testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                            batch_size=int(len(idxs_test)/10), shuffle=False)
    return trainloader, validloader, testloader


# In[6]:


criterion = nn.NLLLoss().to(device)


# In[7]:


if args.optimizer == 'sgd':
    optimizer_global = torch.optim.SGD(global_model.parameters(), lr=args.lr, weight_decay=1e-4)
elif args.optimizer == 'adam':
    optimizer_global = torch.optim.Adam(global_model.parameters(), lr=args.lr, weight_decay=1e-4)


# In[8]:


def inference(model, dataloader):
    """ Returns the inference accuracy and loss.
    """

    model.eval()
    total, correct = 0.0, 0.0
    loss = list()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += [batch_loss.item()]

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = sum(loss)/len(loss)
    return accuracy, loss


# In[9]:


# prepare dataloaders for each client
trainloader, validloader, testloader = dict(), dict(), dict()
trainloader_iterator, validloader_iterator, testloader_iterator = dict(), dict(), dict()
local_acc, local_loss = dict(), dict()

for idx in range(args.num_users):
    idxs = list(user_groups[idx])
    trainloader[idx], validloader[idx], testloader[idx] = train_val_test(train_dataset, idxs)
    trainloader_iterator[idx] = iter(trainloader[idx])
    validloader_iterator[idx] = iter(validloader[idx])
    testloader_iterator[idx]  = iter(testloader[idx])
    local_acc[idx] = list()
    local_loss[idx] = list()
    
global_acc = []
global_loss = []

num_users = max(int(args.frac * args.num_users), 1)
print('activate users %d/%d'%(num_users, args.num_users))

for global_iter in range(100):
   
    idxs_users = np.random.choice(range(args.num_users), num_users, replace=False)
    
    # save local grad gradient
    local_grads = dict()
    for name, params in global_model.named_parameters():
        if params.requires_grad:
            local_grads[name] = torch.zeros_like(params.data)

    start_time = time.time()
    for idx in idxs_users:
        # load single mini-batch
        try:
            images, labels = next(trainloader_iterator[idx])
        except StopIteration:
            trainloader_iterator[idx] = iter(trainloader[idx])
            images, labels = next(trainloader_iterator[idx])

        # create local model
        model = copy.deepcopy(global_model)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # train local model
        images, labels = images.to(device), labels.to(device)
        model.train()
        model.zero_grad()
        log_probs = model(images)
        loss = criterion(log_probs, labels)
        loss.backward()
        
        for name, params in model.named_parameters():
            if params.requires_grad:
                if global_iter>0:
                    params.grad.data = params.grad.data*args.gamma + snap_grads[name]*(1-args.gamma)
                local_grads[name] += params.grad.data
        optimizer.step() 
        
        # test local model
        acc, loss = inference(model, testloader[idx])
        local_acc[idx] += [acc]
        local_loss[idx] += [loss]
        # print('local %d, acc %f, loss %f'%(idx, acc, loss))
        
    print('time %d: %f'%(global_iter, time.time()-start_time))
    
    # make sure global_model has grad.data
    global_model.train()
    if global_iter==0:
        log_probs = global_model(images)
        loss = criterion(log_probs, labels)
        loss.backward()
    
    optimizer_global.zero_grad()
    for name, params in global_model.named_parameters():
        if params.requires_grad:
            local_grads[name] = local_grads[name]/args.num_users
            params.grad = local_grads[name]
    snap_grads = copy.deepcopy(local_grads)
    
    # test global model
    list_acc, list_loss = [], [] 
    for idx in idxs_users:
        acc, loss = inference(global_model, testloader[idx])
        list_acc.append(acc)
        list_loss.append(loss)
    list_acc = sum(list_acc)/len(list_acc)
    list_loss = sum(list_loss)/len(list_loss)
    global_acc += [list_acc]
    global_loss += [list_loss]
    
    print('global %d, acc %f, loss %f'%(global_iter, list_acc, list_loss))
    optimizer_global.step()


# In[10]:


import pickle
with open('results_gamma_%.1f'%(args.gamma), 'wb') as f:
    pickle.dump([global_acc, global_loss, local_acc, local_loss], f)


# In[15]:


import matplotlib.pyplot as plt
##################################
fig, axs = plt.subplots()

gamma_list = [0.1, 0.5, 1]
for gamma in gamma_list:
    with open('results_gamma_%.1f'%(gamma), 'rb') as f:
        [global_acc, global_loss, local_acc, local_loss] = pickle.load(f)
    y = global_acc
    x = np.arange(len(y))
    axs.plot(x,y,label='gamma=%.1f'%(gamma))
    
plt.title('Glocal accuracy / Global communication')
axs.set_xlabel('Global communication')
axs.set_ylabel('Accuracy')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('global_acc.pdf')
plt.close()

##################################
fig, axs = plt.subplots()

for gamma in gamma_list:
    with open('results_gamma_%.1f'%(gamma), 'rb') as f:
        [global_acc, global_loss, local_acc, local_loss] = pickle.load(f)
    y = global_loss
    x = np.arange(len(y))
    axs.plot(x,y,label='gamma=%.1f'%(gamma))
    
plt.title('Global loss / Global communication')
axs.set_xlabel('Global communication')
axs.set_ylabel('Loss')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('global_loss.pdf')
plt.close()

##################################
fig, axs = plt.subplots()

for gamma in gamma_list:
    with open('results_gamma_%.1f'%(gamma), 'rb') as f:
        [global_acc, global_loss, local_acc, local_loss] = pickle.load(f)
        
    acc = []
    for idx in range(args.num_users):
        acc.append(local_acc[idx])
    acc = np.mean(acc,0)
        
    y = acc
    x = np.arange(len(y))
    axs.plot(x,y,label='gamma=%.1f'%(gamma))
    
plt.title('Average local accuracy / Global communication')
axs.set_xlabel('Global communication')
axs.set_ylabel('Accuracy')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('local_acc.pdf')
plt.close()

##################################
fig, axs = plt.subplots()

for gamma in gamma_list:
    with open('results_gamma_%.1f'%(gamma), 'rb') as f:
        [global_acc, global_loss, local_acc, local_loss] = pickle.load(f)
        
    loss = []
    for idx in range(args.num_users):
        loss.append(local_loss[idx])
    loss = np.mean(loss,0)
        
    y = loss
    x = np.arange(len(y))
    axs.plot(x,y,label='gamma=%.1f'%(gamma))
    
plt.title('Average local loss / Global communication')
axs.set_xlabel('Global communication')
axs.set_ylabel('Loss')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('local_loss.pdf')
plt.close()


# In[12]:


acc = []
for idx in range(args.num_users):
    acc.append(local_acc[idx])
np.mean(acc,0)


# In[13]:


acc = np.mean(acc,0)


# In[ ]:




