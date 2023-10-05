#!/usr/bin/env python
# coding: utf-8

# In[1]:
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, mean_squared_error, precision_recall_curve
from sklearn.metrics import precision_recall_curve
from datetime import datetime
import os
import wandb
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import segmentation_models_pytorch as smp
import albumentations as A  
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from torchvision.transforms import ToTensor
import yaml

# In[2]:

with open("/home/dataset/config_unet.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)



myobj = datetime.now()
now_time = myobj.strftime("%d%m%Y_%H%M")
now_time2 = myobj.strftime("%d/%m/%Y %H%M")
log_txt_name = "/home/log_unet/log_unet_" + now_time + ".txt"
with open(log_txt_name, 'a') as plik:
    plik.write('Start learning: ' + now_time2 + '\n')
    plik.write('Parameteres: '+"\n Epoch: "+ str(data_loaded['epoch']) + 
    "\n Batch size: " + str(data_loaded['batch_size']) + 
    "\n Input classes: " + str(data_loaded['input_classes']) +
    "\n Input channels: " + str(data_loaded['input_channels']) +
    "\n Input image size: " + str(data_loaded['img_h']) + " x " + str(data_loaded['img_w']) + '\n')


os.system('wandb login')
a = wandb.init(
    # set the wandb project where this run will be logged
    project="segmentation",
    config=data_loaded
)


# In[3]:

def make_folder(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print("Utworzono folder:", path)
        except OSError as e:
            print("Błąd podczas tworzenia folderu:", path)
            print(e)
    else:
        print("Folder już istnieje:", path)

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)


# In[4]:


class SegmentationDataset(Dataset):
    def __init__(self, input_dir, is_train, transform=None):
        self.input_dir  = input_dir
        self.transform  = transform
        self.is_train = is_train
        if is_train == True:
            path = input_dir + '/images/train'
            self.images = os.listdir(path)
        else:
            path = input_dir + '/images/val'
            self.images = os.listdir(path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.is_train == True:
            img_path = self.input_dir +'/images/train/'+ self.images[index]
            mask_path = self.input_dir +'/masks/train/'+ self.images[index]
        else:
            img_path = self.input_dir +'/images/val/'+ self.images[index]
            mask_path = self.input_dir +'/masks/val/'+ self.images[index]
            
        img         = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) /255
        mask        = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) 

        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)
            img   = augmentations["image"]
            mask  = augmentations["mask"]
        
        return img, mask


# In[10]:


TRAIN_INP_DIR = data_loaded['dataset']
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = data_loaded['learning_rate']
BATCH_SIZE    = data_loaded['batch_size']
NUM_EPOCHS    = data_loaded['epoch']
IMAGE_HEIGHT  = data_loaded['img_h']  
IMAGE_WIDTH   = data_loaded['img_w']
PATIENCE      = data_loaded['patience']


train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=0, p=0.5),
        ToTensorV2(),
    ],
)

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(),
    ],
)


# In[11]:


def get_loaders( inp_dir, batch_size,
			     train_transform, val_tranform ):
    
    train_ds     = SegmentationDataset( input_dir=inp_dir,
                            is_train=True, transform=train_transform)

    train_loader = DataLoader( train_ds, batch_size=BATCH_SIZE, shuffle=True )

    val_ds       = SegmentationDataset( input_dir=inp_dir, 
                            is_train=False, transform=val_transform)

    val_loader   = DataLoader( val_ds, batch_size=BATCH_SIZE, shuffle=True  )

    return train_loader, val_loader


# In[12]:


train_loader, val_loader = get_loaders( TRAIN_INP_DIR, 
                            BATCH_SIZE,  train_transform, val_transform)


# In[13]:
params = dict(
    dropout=0.5,
    classes=data_loaded['input_classes']
)

model = smp.Unet(encoder_name='resnet18', in_channels=data_loaded['input_channels'], aux_params=params).to(DEVICE)
loss_fn   = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# In[14]:
five_more = 0
f = False
counter = 0
lowest_val_loss = 100
f = False
loss_rem_val = 0
loss_act_val = 0
val_loss = 0


val_dice_scores = []
val_precisions = []
val_recalls = []
val_f1_scores = []
val_roc_aucs = []
val_mses = []
val_precisions_recall = []
val_avg_precision = []
val_avg_recall = []
epoch_act = 0

for epoch in range(NUM_EPOCHS):
    epoch_act = epoch
    device = "cuda"
    print('########################## epoch: ' + str(epoch))
    loop = tqdm(train_loader)
    model.train()
    for batch_idx, (image, mask) in enumerate(loop):
        image = image.to(device=DEVICE)
        mask = mask.to(device=DEVICE)
        mask = mask.type(torch.long)

        predictions = model(image)
        predictions = predictions[0]  # Extract the tensor from the tuple

        loss = loss_fn(predictions, mask)
        metrics = {"train/train_loss": loss, "epoch": epoch}


        model.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        wandb.log(metrics)

        loss_act_val = loss.item()


    if (f==True and loss_act_val < loss_rem_val):
        loss_rem_val = loss_act_val
        torch.save(model, 'unet_model.pt')
    elif f==False:
        torch.save(model, 'unet_model.pt')
        loss_rem_val = loss_act_val
        f = True

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    val_avg_precision = 0
    val_avg_recall = 0
    model.eval()

    with torch.no_grad():
      for img, mask in tqdm(val_loader):
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          img = img.to(device)
          mask = mask.to(device)
          mask2 = mask.type(torch.long)

          preds_tuple = model(img)
          preds = preds_tuple[0]

          loss = loss_fn(preds, mask2)
          val_loss = loss.item()
          
          num_correct += (preds.argmax(dim=1) == mask).sum()
          num_pixels += torch.numel(mask)

           

    if num_pixels > 0:
        a = num_correct / num_pixels * 100
    else:
        a = 0

    metrics = {

        "epoch": epoch,
        "val/val_loss": val_loss,
        "counter": counter
    }
    wandb.log(metrics)


    # Early stop
    print("Val_loss",val_loss)
    if val_loss<lowest_val_loss:
        counter = 0
        lowest_val_loss = val_loss
    else:
        counter += 1 
        if counter >= PATIENCE:
            with open(log_txt_name, 'a') as plik:
                plik.write('Conuter: '+ str(counter)+ '\n')
                plik.write('Patience: '+ str(PATIENCE)+ '\n')
                plik.write('Lowest lost: '+ str(lowest_val_loss)+ '\n')
            break

    print("Counter", counter)



myobj = datetime.now()
now_time = myobj.strftime("%d/%m/%Y %H:%M:%S")
with open(log_txt_name, 'a') as plik:
    plik.write('Learning ended: ' + now_time + ' at epoch '+ str(epoch_act) + '\n')

