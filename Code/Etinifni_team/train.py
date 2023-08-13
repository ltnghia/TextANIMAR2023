from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import clip
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import sys
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = np.array(img)
        img = img[:, :, 0:3]
        img = Image.fromarray(img)
        return img

def get_contain_csv(path):
    rows = pd.read_csv(path)
    return rows

class SHRECDataset(torch.utils.data.Dataset):
    def __init__(self, labels, dir_path_view_folder, dir_path_text, transforms):
        """
        dir_sketch: path to folder contain sketch (.png files)
        """
        self.labels = get_contain_csv(labels)
        self.dir_path_view_folder = dir_path_view_folder
        self.transforms = transforms
        self.CSV_text = pd.read_csv(dir_path_text, index_col = 0)
    def __getitem__(self, idx):
        item = {}
        text_id, folder_name = self.labels.loc[idx]

        folder_path = os.path.join(self.dir_path_view_folder, str(folder_name))
        res = custom_loader(folder_path)
        res = self.transforms(res)

        item['obj'] = res
        item['text'] = clip.tokenize(self.CSV_text.loc[text_id].values[0])[0]

        return item
    def __len__(self):
        return len(self.labels)

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def exclude(n): return "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
def include(n): return not exclude(n)

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss

def forward(model, batch):
    list_image, list_txt = batch["obj"], batch["text"]
    images = list_image.to(device)
    texts = list_txt.to(device)

    text_embeddings = model.encode_text(texts)
    image_embeddings = model.encode_image(images)

    # normalized features
    image_embeddings = image_embeddings / \
        image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings / \
        text_embeddings.norm(dim=-1, keepdim=True)

    # calculate targets
    logit_scale = model.logit_scale.exp()
    logits = (image_embeddings @ text_embeddings.T) * logit_scale

    # targets = torch.arange(len(list_txt), dtype=torch.long, device=device)

    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax(
        (images_similarity + texts_similarity) / 2 * logit_scale, dim=-1
    )

    texts_loss = cross_entropy(logits, targets, reduction='none').mean()
    images_loss = cross_entropy(logits.T, targets.T, reduction='none').mean()

    # images_loss = loss_txt(logits, targets)
    # texts_loss = loss_txt(logits.T, targets)
    total_loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
    return total_loss

def eval():
    all_loss = 0
    total = len(test_dataloader)
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            total_loss = forward(model, batch)
            all_loss += total_loss.cpu().detach().numpy()
    print("Test Loss:", all_loss/total)
    return all_loss/total
clip.available_models()

model_name = "ViT-B/16"
model, preprocess = clip.load(model_name, device=device, jit=False)
if device == "cpu":
    model.float()
else:
    # Actually this line is unnecessary since clip by default already on float16
    clip.model.convert_weights(model)

dataset = SHRECDataset(
    labels = './content/mapped_TextQuery_GT_Train.csv',
    dir_path_view_folder = './content/original/multiviews_v3_rotated',
    dir_path_text = "./content/mapped_TextQuery_Train.csv",
    transforms=preprocess,
)
test_dataset = SHRECDataset(
    labels = './content/mapped_TextQuery_GT_Val.csv',
    dir_path_view_folder = './content/original/multiviews_v3_rotated',
    dir_path_text = "./content/mapped_TextQuery_Val.csv",
    transforms=preprocess,
)

BATCH_SIZE = 48
EPOCH = 100
save_file = './content/weight_text_2D_method_v4.pt'
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

named_parameters = list(model.named_parameters())
gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

optimizer = optim.AdamW(
    [
        {"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": 0.01},
    ],
    lr=1e-6,
    betas=(0.9, 0.98),
    eps=1e-6,
)

total_steps = len(train_dataloader) * EPOCH
scheduler = cosine_lr(optimizer, 1e-5, 10000, EPOCH)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
best_loss = 10
current_epoch = -1
early_stopping = 0

eval()
num_batches_per_epoch = len(train_dataloader)
save_step = 1100
for epoch in range(EPOCH):
    if epoch <= current_epoch:
        continue
    print("Epoch:", epoch)
    all_loss = 0
    total = len(train_dataloader)
    num = 0
    model.train()
    with tqdm(train_dataloader, total=total) as pbar:
        for batch in pbar:
            num += 1
            step = num_batches_per_epoch * epoch + num
            scheduler(step)
            optimizer.zero_grad()

            total_loss = forward(model, batch)
            all_loss += total_loss.cpu().detach().numpy()
            pbar.set_description(
                f"Loss: {total_loss.cpu().detach().numpy():0.4f}, Average: {all_loss/num:0.4f}")
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            if num % save_step == 0:
                test_loss = eval()
                if test_loss < best_loss:
                    early_stopping = 0
                    best_loss = test_loss
                    print("Best Loss! Saving model.")
                    torch.save({"state_dict": model.state_dict()}, save_file)
                else:
                    early_stopping += 1
                    if early_stopping > 10:
                        print("Loss doesn't decrease. Quitting.")
                        sys.exit()

            model.logit_scale.data = torch.clamp(
                model.logit_scale.data, 0, 4.6052)

    test_loss = eval()

    if test_loss < best_loss:
        early_stopping = 0
        best_loss = test_loss
        print("Best Loss! Saving model.")
        torch.save({"state_dict": model.state_dict()}, save_file)
    else:
        early_stopping += 1
        if early_stopping > 10:
            print("Loss doesn't decrease. Quitting.")
            sys.exit()