import os
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
import glob
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms
from ultralytics import YOLO
import glob
import os
import cv2
from PIL import Image

def make_folder(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print("Folder created:", path)
        except OSError as e:
            print("Error:", path)
            print(e)
    else:
        pass
        # print("Folder already exists:", path)

model = YOLO('/home/chaneu/remodel/yolo_model_3d.pt')
path_in_folder = '/home/chaneu/remodel/dataset/'
images = glob.glob(os.path.join(path_in_folder,  "images_4/test/*.png"))
path_out_folder = '/home/chaneu/remodel/yolo3d'
path_out_folder_image = '/home/chaneu/remodel/yolo3d/images'
path_out_folder_mask = '/home/chaneu/remodel/yolo3d/masks'
path_out_folder_pred = '/home/chaneu/remodel/yolo3d/preds'
make_folder(path_out_folder)
make_folder(path_out_folder_image)
make_folder(path_out_folder_mask)
make_folder(path_out_folder_pred)


dic = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255)}
dic_names = {0: 'conn_a', 1: 'root', 2: 'box', 3: 'bag_a', 4: 'bag_b'}


for i in images:
    img_to_model = cv2.imread(i, cv2.IMREAD_UNCHANGED)
    photo_num = i.split('/')[-1]
    rgb_image_path = path_in_folder + 'images/test/' + photo_num
    rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_UNCHANGED)
    img_path = path_in_folder + 'images_4/test/' + photo_num
    label_path = path_in_folder + 'masks/test/' + photo_num
    end_images = path_out_folder + '/images/' + photo_num
    end_masks = path_out_folder + '/masks/'  + photo_num
    end_preds = path_out_folder + '/preds/'  + photo_num
    
    pred = model(img_to_model)
    H, W, _ = img_to_model.shape
    dst = img_to_model

    img_copy = rgb_image
    copy_image = rgb_image
    out_mask = np.zeros_like(copy_image)
    
    for se in pred:
        if se:
            clist = se.boxes.cls
            clss = []
            masks = []
            clean = np.zeros_like(rgb_image)
            clean_bouding_box = np.zeros_like(rgb_image)
            contour_list = []

            for cno in clist:
                clss.append(int(cno))

            for j, mask in enumerate(se.masks.data):
                mask = mask.cpu().numpy() * 255
                mask = cv2.resize(mask,(W,H))
                value = dic[clss[j]]

                mask = mask[:,:, np.newaxis]
                
                end_mask = cv2.merge((mask,mask,mask)) #3 channel mask
                out_1 = (np.where(end_mask>0, value, end_mask)).astype(np.uint8)
                out_mask = (np.where(end_mask>0, clss[j]+1, out_mask)).astype(np.uint8)
                copy_image = np.where(out_1>0, out_1, copy_image)
               

                #serching contour to draw boudingboxes
                _, mask0 = cv2.threshold(mask.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(mask0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                max_contour = max(contours, key = cv2.contourArea)
                rect = cv2.boundingRect(max_contour)
                contour_list.append((rect,j))

            for rect, j in contour_list:
                x,y,w,h = rect
                cv2.rectangle(copy_image,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(copy_image, dic_names[clss[j]], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)



            dst = cv2.addWeighted(copy_image, 0.7, img_copy, 0.3, 0.0)
        
        
        label = Image.open(label_path).convert("L")
        label_resized = label
        label_resized_0 = np.array(label_resized, dtype=np.uint8)

        cv2.imwrite(end_images, dst) #image with boundingboxes
        cv2.imwrite(end_masks, label_resized_0) #true mask
        cv2.imwrite(end_preds, out_mask) #predicted mask
  
