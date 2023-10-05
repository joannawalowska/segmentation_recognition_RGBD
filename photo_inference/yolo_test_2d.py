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

model = YOLO('/home/chaneu/remodel/yolo_model.pt')
path_in_folder = '/home/chaneu/remodel/dataset/'
images = glob.glob(os.path.join(path_in_folder,  "images/test/*.png"))
path_out_folder = '/home/chaneu/remodel/yolo'
path_out_folder_image = '/home/chaneu/remodel/yolo/images'
path_out_folder_mask = '/home/chaneu/remodel/yolo/masks'
path_out_folder_pred = '/home/chaneu/remodel/yolo/preds'
make_folder(path_out_folder)
make_folder(path_out_folder_image)
make_folder(path_out_folder_mask)
make_folder(path_out_folder_pred)



dic = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255)}
dic_names = {0: 'conn_a', 1: 'root', 2: 'box', 3: 'bag_a', 4: 'bag_b'}

for i in images:
    img_to_model = cv2.imread(i)
    photo_num = i.split('/')[-1]
    img_path = path_in_folder + 'images/test/' + photo_num
    label_path = path_in_folder + 'masks/test/' + photo_num
    end_images = path_out_folder + '/images/' + photo_num
    end_masks = path_out_folder + '/masks/'  + photo_num
    end_preds = path_out_folder + '/preds/'  + photo_num

    
    pred = model(img_to_model)
    H, W, _ = img_to_model.shape
    dst = img_to_model

    img_copy = img_to_model.copy()
    copy_image = img_to_model.copy()
    out_mask = np.zeros_like(copy_image)
  
    
    for se in pred:
        if se:
            clist = se.boxes.cls
            clss = []
            masks = []
            clean = np.zeros_like(img_to_model)
            clean_bouding_box = np.zeros_like(img_to_model)
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
                img_to_model = np.where(out_1>0, out_1, img_to_model)
              

                #serching contour to draw boudingboxes
                _, mask0 = cv2.threshold(mask.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(mask0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                max_contour = max(contours, key = cv2.contourArea)
                rect = cv2.boundingRect(max_contour)
                
                contour_list.append((rect,j))

            for rect, j in contour_list:
                x,y,w,h = rect
                cv2.rectangle(img_to_model,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img_to_model, dic_names[clss[j]], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
           
            dst = cv2.addWeighted(img_to_model, 0.7, img_copy, 0.3, 0.0)
        
        
        label = Image.open(label_path).convert("L")
        label_resized = label
        label_resized_0 = np.array(label_resized, dtype=np.uint8)

        cv2.imwrite(end_images, dst) #image with boundingboxes
        cv2.imwrite(end_masks, label_resized_0) #true mask
        cv2.imwrite(end_preds, out_mask) #predicted mask
  
