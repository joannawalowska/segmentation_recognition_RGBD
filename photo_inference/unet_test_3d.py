import os
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from torchvision.transforms import ToTensor
import glob

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

path_in_folder = '/home/chaneu/dataset/'
images = glob.glob(os.path.join(path_in_folder,  "images/test/*.png"))
path_out_folder = '/home/chaneu/unet3d'
path_out_folder_image = '/home/chaneu/unet3d/images'
path_out_folder_mask = '/home/chaneu/unet3d/masks'
path_out_folder_pred = '/home/chaneu/unet3d/preds'
make_folder(path_out_folder)
make_folder(path_out_folder_image)
make_folder(path_out_folder_mask)
make_folder(path_out_folder_pred)
model_unet = torch.load('/home/chaneu/unet_model.pt')
model_tiff = torch.load('/home/chaneu/one_ch_model.pt')
softmax = nn.Softmax(dim=1)
DEVICE = 'cuda'

print()

dic = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0), 5: (255, 0, 255)}
dic_names = {1: 'conn_a', 2: 'root', 3: 'box', 4: 'bag_a', 5: 'bag_b'}

for num, im in enumerate(images):
    photo_num = im.split('/')[-1]
    img_path = path_in_folder + 'images/test/' + photo_num
    tiff_path = path_in_folder + 'images_1/test/' + photo_num[:-4] + '.tiff'
    label_path = path_in_folder + 'masks/test/' + photo_num
    end_images = path_out_folder + '/images/' + photo_num
    end_masks = path_out_folder + '/masks/'  + photo_num
    end_preds = path_out_folder + '/preds/'  + photo_num

    convert_tensor = transforms.ToTensor()
    resize = transforms.Resize(size = (512,768),interpolation =  cv2.INTER_NEAREST)
    label = Image.open(label_path).convert("L")
    label_resized = resize(label)
    resized_image = resize(Image.open(img_path).convert("RGB"))
    resized_tiff = resize(Image.open(tiff_path))

    img2 = convert_tensor(np.array(resized_image, dtype=np.float32) /255)
    tiff = convert_tensor(np.array(resized_tiff, dtype=np.float32))

    #RGB image
    img2 = img2.to(device=DEVICE)
    img2 = img2.unsqueeze(0)
    pred = model_unet(img2)
    tmp = torch.argmax(softmax(pred[0]),axis=1)
    tmp = tmp.squeeze(0)
    prediction = np.array(tmp.cpu())
    pred_mask = prediction[:,:, np.newaxis]
    pred_mask = np.uint8(pred_mask)
    copy_pred = cv2.merge((pred_mask,pred_mask,pred_mask))
    cp = np.where(copy_pred==1, (255, 0, 0), copy_pred)
    cp = np.where(cp==2, (0, 255, 0), cp)
    cp = np.where(cp==3, (0, 0, 255), cp)
    cp = np.where(cp==4, (255, 255, 0), cp)
    cp = np.where(cp==5, (255, 0, 255), cp)
    cp = cp.astype(np.uint8)

    #D image
    tiff = tiff.to(device=DEVICE)
    tiff = tiff.unsqueeze(0)
    pred = model_tiff(tiff)
    tmp2 = torch.argmax(softmax(pred[0]),axis=1)
    tmp2 = tmp2.squeeze(0)
    prediction_tiff = np.array(tmp2.cpu())
    pred_mask_tiff = prediction_tiff[:,:, np.newaxis]
    pred_mask_tiff = np.uint8(pred_mask_tiff)
    pred_mask_tiff = np.where(pred_mask_tiff>5, 0, pred_mask_tiff)
    copy_pred_tiff = cv2.merge((pred_mask_tiff,pred_mask_tiff,pred_mask_tiff))
    cp_tiff = np.where(copy_pred_tiff==1, (255, 0, 0), copy_pred_tiff)
    cp_tiff = np.where(cp_tiff==2, (0, 255, 0), cp_tiff)
    cp_tiff = np.where(cp_tiff==3, (0, 0, 255), cp_tiff)
    cp_tiff = np.where(cp_tiff==4, (255, 255, 0), cp_tiff)
    cp_tiff = np.where(cp_tiff==5, (255, 0, 255), cp_tiff)
    cp_tiff = cp_tiff.astype(np.uint8)

    #RGBD image
    output = np.where(copy_pred==0, copy_pred_tiff , copy_pred) #mask
    out = np.where(output==1, (255, 0, 0), output)
    out = np.where(out==2, (0, 255, 0), out)
    out = np.where(out==3, (0, 0, 255), out)
    out = np.where(out==4, (255, 255, 0), out)
    out = np.where(out==5, (255, 0, 255), out)
    out = out.astype(np.uint8) #color mask

    image = cv2.imread(img_path)
    resized_image = cv2.resize(image, (768, 512), interpolation = cv2.INTER_AREA)
    copy_image = resized_image.copy()
    copy_image_tiff = resized_image.copy()

    contour_list = []

    #Get all contours
    mask_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    for b in range(1,6):
      tmp = np.where(mask_gray==b, b, 0).astype(np.uint8)
      _, mask = cv2.threshold(tmp, 1, 255, cv2.THRESH_BINARY)
      contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      for i in contours:
        contour_list.append(i)


    with_labels = cv2.add(out, copy_image)
    dst = cv2.addWeighted(with_labels, 0.7, copy_image, 0.3, 0.0)
    
    for c in contour_list:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),1)
        counter = 0
        while True and counter<1000: #serching point inside contour
            random_point = (np.random.randint(x, x + w), np.random.randint(y, y + h))
            counter += 1
            if cv2.pointPolygonTest(c, random_point, False) > 0:
                value = output[random_point[1], random_point[0]][0] #value inside contour
                if value > 0:
                  cv2.putText(dst, dic_names[int(value)], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                  break

    label_resized_0 = np.array(label_resized, dtype=np.uint8)
 
    cv2.imwrite(end_images, dst) #image with boundingboxes
    cv2.imwrite(end_masks, label_resized_0) #true mask
    cv2.imwrite(end_preds, output) #predicted mask





