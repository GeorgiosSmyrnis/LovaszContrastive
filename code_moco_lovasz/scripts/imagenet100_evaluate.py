#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import sys
import os
from tqdm import tqdm
# Step 1: Load the model:

model_pth = sys.argv[1]

loaded = torch.load(model_pth, map_location='cpu')
model = models.__dict__[loaded['arch']](num_classes=100)
state_dict = {k.replace('module.', ''): v for k, v in loaded['state_dict'].items()}

model.load_state_dict(state_dict)
model = model.eval()
model = model.cuda()

# Step 2 load the validation set
BASE_DIR = '/path/to/data'
valdir = os.path.join(BASE_DIR, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False,
                                         num_workers=8, pin_memory=True)



# Step 3: Evaluate accuracy
count = 0
correct = 0
with torch.no_grad():
    for x, y in tqdm(dataloader):
        x, y = x.cuda(), y.cuda()
        count += y.numel()
        correct += (model(x).max(dim=1)[1] == y).sum()

print(count, correct, correct / count)