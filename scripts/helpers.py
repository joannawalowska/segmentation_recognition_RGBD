#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor


# ## Find polygons on mask and make txt file

# In[23]:


def find_polygons_save_txt_file(input_dir, output_dir, photo_dir):
    for j in os.listdir(input_dir):
        mask_value = []
        mask_polygon = []
        polygons = []
        image = None
        image_path = os.path.join(input_dir, j)
        photo_path = os.path.join(photo_dir, j)
        photo = cv2.imread(photo_path)
        # load the binary mask and get its contours
        mask_0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask_1 =  np.uint8(np.where(mask_0>0, 255, mask_0))
        
        mask_00 = cv2.imread(image_path)  
        mask_2 =  np.uint8(np.where(mask_00>0, [255, 255, 255],[0, 0, 0]))
        mask_NOT = np.uint8(mask_00)
        
        _, mask = cv2.threshold(mask_1, 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for random_contour in contours: #contours on actual photo
            cv2.drawContours(photo, [random_contour], -1, (0,100,0), 3) #paint contour on photo
            x, y, w, h = cv2.boundingRect(random_contour) #boundingbox around the contour
            while True: #search point inside contours
                random_point = (np.random.randint(x, x + w), np.random.randint(y, y + h))
                if cv2.pointPolygonTest(random_contour, random_point, False) >= 0:
                    value = mask_NOT[random_point[1], random_point[0]][0] #get mask value from the point of the contour
                    mask_value.append(value-1) #add mask value to list
                    image = cv2.putText(photo, str(value-1), random_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    break  

            polygon = []
            H, W = mask_1.shape
            for point in random_contour: #get points from contours
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
                
            if(len(polygon) % 2):
                print("ERROR", j)
                
            mask_polygon.append(polygon)
        with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f: # zapis punktów do pliku txt
            for num, polygon in enumerate(mask_polygon):
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon)-1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write(str(mask_value[num]))
                        f.write(' {} '.format(p))
                    else:
                        f.write('{} '.format(p))

            f.close()
            name_out = os.path.join(output_dir, j)
            cv2.imwrite(name_out, image)
            
    print("DONE")
    


# ## Creating masks 

# In[19]:


files_mask = glob.glob('/media/chaneu/Atlantyda/zuzki/mask/*')
files_photo = glob.glob('/media/chaneu/Atlantyda/zuzki/png/*')
print(len(files_mask))
print(len(files_photo))
phot_bef = files_mask[0].split("-")[1]
mask_bef = []
for i in files_mask:
    sp = i.split("-")
    orig_photo_name = files_photo[int(sp[1])-101].split("/")
    photo = cv2.imread(files_photo[int(sp[1])-101])
    tiff_name = '/media/chaneu/Atlantyda/zuzki/tiff/' + orig_photo_name[-1][:-3] +'tiff'
    mask = cv2.imread(i)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    name = '/media/chaneu/Atlantyda/zuzki/out/' + sp[-2] + '/' + sp[-1][:1] + '-' + orig_photo_name[-1] 
    name_p = '/media/chaneu/Atlantyda/zuzki/out_p/' + sp[-2] + '/' + sp[-1][:1] + '-' + orig_photo_name[-1] 
    name_t = '/media/chaneu/Atlantyda/zuzki/out_t/' + sp[-2] + '/' + sp[-1][:1] + '-' + orig_photo_name[-1][:-3] +'tiff'
    name_3d = '/media/chaneu/Atlantyda/zuzki/out_3d/' + sp[-2] + '/' + sp[-1][:1] + '-' + orig_photo_name[-1] 

    if sp[-2]=='conn_a':
        out = np.where(mask > 0, [1, 1, 1], mask)
    elif sp[-2]=='root':
        out = np.where(mask > 0, [2, 2, 2], mask)
    elif sp[-2]=='box':
        out = np.where(mask > 0, [3, 3, 3], mask)
    elif sp[-2]=='bag_a':
        out = np.where(mask > 0, [4, 4, 4], mask)
    elif sp[-2]=='bag_b':
        out = np.where(mask > 0, [5, 5, 5], mask)
    cv2.imwrite(name, out)
    cv2.imwrite(name_p, photo)
    get_ipython().system('cp $tiff_name $name_t')

    tiff_image = Image.open(name_t)
    png_image = Image.open(name_p)
    tiff_tensor = ToTensor()(tiff_image)
    png_tensor = ToTensor()(png_image)
    merged_tensor = torch.cat((tiff_tensor, png_tensor), dim=0)
    merged_image = ToPILImage()(merged_tensor)
    merged_image.save(name_3d)

    #TEST IF THE PHOTO COMBINATION IS RIGHT
    h = '/media/chaneu/Atlantyda/zuzki/valid/'+ sp[-2] + '-' + sp[-1][:1] + '-' + orig_photo_name[-1]
    dst = cv2.addWeighted(photo, 0.5, mask.astype(np.uint8), 0.5, 0.0)
    cv2.imwrite(h, dst)
#     break
print("DONE")



# ## Merging masks in one

# In[3]:


files_mask = glob.glob('/media/chaneu/Atlantyda/zuzki/mask/*')
files_photo = sorted(glob.glob('/media/chaneu/Atlantyda/zuzki/png/*'))
print(files_photo[0])
print(len(files_mask))

photo_bef = files_mask[0].split("-")[1]
end_list = []
for i in range(len(files_photo)):
    end_list.append([None, [], None, None, 
                     None, None, None]) # (photo, [mask list], photo path, mask path, tiff path in, tiff path put, 3d image path out)

for i in files_mask:
    sp = i.split("-")
    photo_num = int(sp[1])-1
    orig_photo_name = files_photo[photo_num].split("/")
    
    photo = cv2.imread(files_photo[photo_num])
    name = '/media/chaneu/Atlantyda/zuzki/out_one/' + orig_photo_name[-1] 
    name_p = '/media/chaneu/Atlantyda/zuzki/out_one_p/' + orig_photo_name[-1] 
    name_t = '/media/chaneu/Atlantyda/zuzki/out_one_t/' + orig_photo_name[-1][:-3] + 'tiff'
    name_t_in = '/media/chaneu/Atlantyda/zuzki/tiff/' + orig_photo_name[-1][:-3] + 'tiff'
    name_3d = '/media/chaneu/Atlantyda/zuzki/out_one_3d/' + orig_photo_name[-1]
    
    mask = cv2.imread(i)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    
    end_list[photo_num][0] = photo
    end_list[photo_num][2] = name_p
    end_list[photo_num][3] = name
    end_list[photo_num][4] = name_t_in
    end_list[photo_num][5] = name_t
    end_list[photo_num][6] = name_3d
    
    if sp[-2]=='conn_a':
        out = np.where(mask > 0, [1, 1, 1], mask)
    elif sp[-2]=='root':
        out = np.where(mask > 0, [2, 2, 2], mask)
    elif sp[-2]=='box':
        out = np.where(mask > 0, [3, 3, 3], mask)
    elif sp[-2]=='bag_a':
        out = np.where(mask > 0, [4, 4, 4], mask)
    elif sp[-2]=='bag_b':
        out = np.where(mask > 0, [5, 5, 5], mask)
        
    end_list[photo_num][1].append(out)
    photo_bef = sp[1]
    
for num, j in enumerate(end_list):
    clear_mask = np.zeros((j[0].shape[0], j[0].shape[1], j[0].shape[2]))

    for i in j[1]:
        clear_mask = np.where(i>0, i, clear_mask)    
        
    test_mask =  np.where(clear_mask>0, [255, 255, 255], clear_mask)
        
    cv2.imwrite(j[3], clear_mask)
    cv2.imwrite(j[2], j[0]) 
#     print(j[4], j[5])
    ni = j[4]
    no = j[5]
    get_ipython().system('cp $ni $no')
    
    tiff_image = Image.open(j[5])
    png_image = Image.open(j[2])
    tiff_tensor = ToTensor()(tiff_image)
    png_tensor = ToTensor()(png_image)
    merged_tensor = torch.cat((tiff_tensor, png_tensor), dim=0)
    merged_image = ToPILImage()(merged_tensor)
    merged_image.save(j[6])
    
    h = '/media/chaneu/Atlantyda/zuzki/valid/' + str(num) + '.png'
    dst = cv2.addWeighted(j[0], 0.5, test_mask.astype(np.uint8), 0.5, 0.0)
    cv2.imwrite(h, dst)
    

print("DONE")
      


# ## Checking mask values

# In[9]:


def check_mask_values(path):
    folder = glob.glob(path + '/*.png')
    folder.sort()
    for i in folder:
        print('----------')
#         print(i)
        mask = cv2.imread(i)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        liczby = []
        new = False
        n = True

        for k in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (gray[k][j] > 0 and n):
                    liczby.append(gray[k][j])
                    n = False
                if (gray[k][j] > 0 and n==False):
                    new=True
                    for i in liczby:
                        if i==gray[k][j]:
                            new=False

                if new and gray[k][j] != 0:
                    liczby.append(gray[k][j])
    
        print("Found contours:", liczby)
        for i in liczby:
            if (i>5):
                print(i)
check_mask_values('/media/chaneu/Atlantyda/na _porthosa_03_08/31.07/dataset/masks/val')
    


# ## Creating mask with given values

# In[7]:


def make_mask_values():
    folder = ['conn_a', 'root', 'box', 'bag_a', 'bag_b'] #name of the folders

    for num1, c in enumerate(folder):
        num = num1 + 1 #value of the mask
        path_wy = '/media/chaneu/Atlantyda/Magi_210623/mask_NOT/' + c + '/'
        path_we = glob.glob('/media/chaneu/Atlantyda/Magi_210623/mask/' + c + '/*.png')

        for i in path_we:
            name = i[-25:]
            print(i)
            filename = os.path.join(path_wy, name)
            maska = cv2.imread(i)
            maska = np.where(maska == [255, 255, 255], [num, num, num], maska)
            cv2.imwrite(filename, maska)


    print("DONE")
make_mask_values()


# In[17]:


# !rm /home/chaneu/Magi/YOLO/labels/train/1677069752.295090_rgb.txt


# ## Check compatibility of files

# In[22]:


def check_compatibility_of_files(folder01, folder02, letters):
    folder1 = glob.glob(folder01 + '/*')
    folder2 = glob.glob(folder02 + '/*')
    
    for i in folder1:
        find = False
        for j in folder2:
            if i[-25:-3] == j[-26:-4]:   
                find = True
                break
        if find:
            continue
        else:
            print(i, "first NO pair")           
            
    for i in folder2:
        find = False
        for j in folder1:
            if i[-26:-4] == j[-25:-3]:
                find = True
                break
        if find:
            continue
        else:
            print(i, "second NO pair")
            get_ipython().system('rm $i')
            
            
    print("checking compatibility end")
                
check_compatibility_of_files("/media/chaneu/Atlantyda/new/out_one_p/", "/media/chaneu/Atlantyda/new/out_one_t/", -25) 
#first folder, second folder, number of letters in name of the file 


# ## Change file names

# In[17]:


# 000000111 000111000
def change_file_name():
    path = '/media/chaneu/Atlantyda/aug/end_augmented_txt/'
    files = glob.glob(path + '*')
    for i in files:
        name = path+ '111' + '111' + i[-7:-4]+ '.txt' 
#         print(name, i)
#         break
#         !mv $i $name
change_file_name()


# ## Resize image

# In[ ]:


def resise_images(image_folder):
    once = True
    folder = ['conn_a', 'conn_b', 'conn_c', 'root', 'conn_d', 'conn_e', 'conn_f','conn_g', 'conn_h', 
              'conn_i', 'conn_j', 'conn_k', 'box', 'bag_red', 'bag_b_s', 'bag_b_b']
    for i in folder:
        path_out = '/home/chaneu/Magi/image_out_768_512/' + i 
        images = glob.glob(image_folder + '/' + i + '/*.png')
        os.makedirs(path_out)
                
        for j in images:
            image = cv2.imread(j)
            image2 = cv2.resize(image, (768, 512), interpolation = cv2.INTER_AREA) #size of the image
            name = j[-25:]
            filename = os.path.join(path_out, name)
            cv2.imwrite(filename, image2)
    print('done')

                            
resise_images('/home/chaneu/Magi/image')

