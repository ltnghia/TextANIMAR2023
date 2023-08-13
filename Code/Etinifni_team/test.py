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



def extract_3D(model):
  linkfolder3D = "./content/original/multiviews_v3_rotated"
  file3D_ID = []
  obj3D = 0
  k = 0
  with torch.no_grad():
    for i in os.listdir(linkfolder3D):
      angle = i.split(".")[0].split("_")[-1]
      if angle == "6" or angle == "8":
        if k == 0:
            image_path = os.path.join(linkfolder3D, i)
            image = custom_loader(image_path)
            image = preprocess(image).unsqueeze(0).to(device)
            obj3D = model.encode_image(image).float()
            obj3D = obj3D/obj3D.norm(dim=-1, keepdim=True)
            file3D_ID.append(i.split('_')[1])
            k = 1
        else:
            image_path = os.path.join(linkfolder3D, i)
            image = custom_loader(image_path)
            image = preprocess(image).unsqueeze(0).to(device)

            embed = model.encode_image(image).float()
            embed = embed/embed.norm(dim=-1, keepdim=True)
            obj3D = torch.concat([obj3D, embed], dim = 0)
            file3D_ID.append(i.split('_')[1])
    print(obj3D.shape)
    print(len(file3D_ID))
  return file3D_ID, obj3D

def find_matches_test(model, embeddings3D, query, real_name_3D):
    text_tok = clip.tokenize(query).to(device)
    embeddingText = model.encode_text(text_tok).float()
    embeddingsText = embeddingText / embeddingText.norm(dim=-1, keepdim=True)
    dot_similarity = embeddingText @ embeddings3D.T

    list_dot = dot_similarity.squeeze(0).tolist()
    dict_dot = {}
    for i in real_name_3D:
      if i not in list(dict_dot.keys()):
        dict_dot[i] = []

    for index, k in enumerate(real_name_3D):
      dict_dot[k].append(list_dot[index])
    for i in list(dict_dot.keys()):
      dict_dot[i] = np.sum(dict_dot[i])
      #dict_dot[i] = np.max(dict_dot[i])
    dict_dot = dict(sorted(dict_dot.items(), key=lambda item: item[1]))
    res = list(dict_dot.keys())
    res.reverse()
    if res[0] == errorObj:
        a = res[0]
        res[0] = res[1]
        res[1] = a
    return res

model_name = "ViT-B/16"
model, preprocess = clip.load(model_name, device=device, jit=False)
if device == "cpu":
    model.float()
else:
    # Actually this line is unnecessary since clip by default already on float16
    clip.model.convert_weights(model)

states = torch.load("./content/weight_text_2D_method_v4.pt")
model.load_state_dict(states['state_dict'])
model.to(device)
model.eval()

mapping = pd.read_csv("./content/original/mapping.txt")
file3D_ID, obj3D = extract_3D(model)
real_name_3D = []
errorObj = -1
for des in file3D_ID:
    real_name_3D.append(mapping.loc[mapping['new_name'] == (des+".obj")]['original'].values[0].split(".")[0])

df_map = pd.read_csv("./content/original/mapping_test_text.csv")
df = pd.read_csv("./content/original/mapped_test_text.csv", index_col=0)

test = df.copy()
for i in test.index:
    S = test.loc[i].values[0]
    S = S.split(" is ")[0]
    test.loc[i][0] = S
test.to_csv('./content/mapped_test_text.csv')

output_test = {}
for query, new_query in zip(df_map["ID"], df_map["newID"]):
    #print(new_query)
    #print(df.loc[new_query].values[0])
    output = find_matches_test(model, obj3D, df.loc[new_query].values[0], real_name_3D)
    output_test[query] = output

out_list = list(output_test.values())
submission = pd.DataFrame(data = out_list)
submission.index = list(output_test.keys())
submission.head()
submission.to_csv("etinifni_TextANIMAR2023.csv", header=False)