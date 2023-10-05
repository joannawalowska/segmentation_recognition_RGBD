#!/usr/bin/env python
# coding: utf-8
import glob
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

os.system('wandb login')
a = wandb.init(
    # set the wandb project where this run will be logged
    project="augumentation"
)


def make_folder(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print("Folder created:", path)
        except OSError as e:
            print("Error", path)
            print(e)
    else:
        print("Folder already exists:", path)

def save_photos(name, end_photo, end_mask_NOT, end_mask):
    photo_name = 'end_augmented_photo/' + name
    mask_NOT_name = 'end_augmented_mask_NOT/' + name
    mask_name = 'end_augmented_mask/' + name
    cv2.imwrite(photo_name, end_photo)
    cv2.imwrite(mask_NOT_name, end_mask_NOT)
    cv2.imwrite(mask_name, end_mask)

def find_polygons_save_txt_file():
    input_dir = 'end_augmented_mask/'
    output_dir = 'end_augmented_txt/'
    for j in os.listdir(input_dir):
        mask_value = []
        mask_polygon = []
        polygons = []
        image_path = os.path.join(input_dir, j)
        # load the binary mask and get its contours
        mask_1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask_2 = cv2.imread(image_path)
        mask_NOT_path = os.path.join('end_augmented_mask_NOT/', j)
        mask_NOT = cv2.imread(mask_NOT_path)
        _, mask = cv2.threshold(mask_1, 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for random_contour in contours: #contours on actual photo
            cv2.drawContours(mask_2, [random_contour], -1, (0,100,0), 3) #paint contour on photo
            x, y, w, h = cv2.boundingRect(random_contour) 
            while True: #search point inside contours
                random_point = (np.random.randint(x, x + w), np.random.randint(y, y + h))
                if cv2.pointPolygonTest(random_contour, random_point, False) >= 0:
                    value = mask_NOT[random_point[1], random_point[0]][0] #get mask value from the point of the contour
                    mask_value.append(value-1) #add mask value to list
                    image = cv2.putText(mask_2, str(value-1), random_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
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


        with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f: # save points to txt file
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
            
    print("DONE")

def gamma_conn_change(img):
    gamma = random.uniform(0.1, 0.9) # 1.1 biggest exposition
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    output = cv2.LUT(img,gamma_table)
    return output

    
def make_augumentations():
    one = True
    log_txt = 'c'
    end_txt_name = 'log_unet.txt'
    out_folder_test = '/home/out_aug_test/'
    make_folder(out_folder_test)
    for b in range(100, 1000):
        path_background = 'background'
        files_background = glob.glob(path_background + '/*')

        path_mask = 'mask'
        files_mask = glob.glob(path_mask + '/*')

        path_image = 'image'
        files_image = glob.glob(path_image + '/*')

        path_mask_NOT = 'mask_NOT'
        files_NOT_full = glob.glob(path_mask_NOT + '/*')
        # Random background
        back_number = random.randint(0, len(files_background)-1)
        background_photo = cv2.imread(files_background[back_number])
        background_photo = cv2.resize(background_photo, (3000, 2500), interpolation = cv2.INTER_AREA)

        # Random number of conns
        conn_num = random.randint(1, 5)
        real_conn_num = conn_num + 1
        #Get conn and mask list to patch
        conn_list = []
        mask_list = []
        mask_NOT_full = []
        # Random conns
        for i in range(conn_num):
            conn_folder = files_mask[random.randint(0, len(files_mask)-1)]
            conn_mask = glob.glob(conn_folder + '/*.png')
            conn = conn_mask[random.randint(0, len(conn_mask)-1)]


            image_folder = str('image'+ conn[4:])
            mask_full_folder = str('mask_NOT'+ conn[4:])
            conn_mask = cv2.imread(conn)
            conn_photo = cv2.imread(image_folder)
            conn_full_mask = cv2.imread(mask_full_folder)

            conn_list.append(conn_photo)
            mask_list.append(conn_mask)
            mask_NOT_full.append(conn_full_mask)
                
        metrics = {"Number": b}
        wandb.log(metrics)
        if len(mask_NOT_full) == len(mask_list) and len(mask_list)==len(conn_list):
            log_txt = str(b) + " lists the same size"
            print(log_txt)
        else:
            log_txt = str(b) + " ERROR, check sizes of lists"
            print(log_txt)
        
        try:
            with open(end_txt_name, 'a') as plik:
                plik.write(log_txt + '\n')
        except IOError as e:
            print("Error:", end_txt_name)
            with open(end_txt_name, 'a') as plik:
                plik.write("Error "+ str(b) + '\n')
            print(e)
            
        cropped_conn_list = []
        cropped_conn_mask = []
        cropped_full_NOT = []

        for num, i in enumerate(mask_list):
            maskBGR = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(maskBGR, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect

            conn_to_cut = cv2.add(conn_list[num], cv2.bitwise_not(i))
            cropped_image_0 = conn_to_cut[y:y+h, x:x+w] 
            cropped_image = gamma_conn_change(cropped_image_0)
            # break
            cropped_mask = i[y:y+h, x:x+w] 

            cropped_NOT_FULL = mask_NOT_full[num][y:y+h, x:x+w] 

            cropped_image = np.where(cropped_image == [255, 255, 255], [0, 0, 0], cropped_image)
            
            cropped_conn_list.append(cropped_image)
            cropped_conn_mask.append(cropped_mask)
            cropped_full_NOT.append(cropped_NOT_FULL)
            
        #Paste conns on image
        padding_x = 30 
        padding_y = 30 
        prev_positions=[]
        image_black = np.zeros((background_photo.shape[0], background_photo.shape[1], 3), dtype=np.uint8) #ostateczny obraz, wtyczki na tle
        image_black_mask = np.zeros((background_photo.shape[0], background_photo.shape[1], 3), dtype=np.uint8) #maska czarno biała
        image_black_label = np.zeros((background_photo.shape[0], background_photo.shape[1], 3), dtype=np.uint8) #maska z różnymi wartościami
        tmp_img = np.zeros((background_photo.shape[0], background_photo.shape[1], 3), dtype=np.uint8)

        for num, small_image in enumerate(cropped_conn_list):
            hah = 0
            h, w = small_image.shape[:2]
            pos = (random.randint(0, background_photo.shape[0]-h), random.randint(0, background_photo.shape[1]-w))
            # Check if image can be pasted
            for prev_pos in prev_positions:
                while abs(pos[0] - prev_pos[0]) < h + padding_y or abs(pos[1] - prev_pos[1]) < w + padding_x:
                    hah = hah+1
                    pos = (random.randint(0, background_photo.shape[0]-h), random.randint(0, background_photo.shape[1]-w))
                    if hah>500: break
            if hah <= 500:
                image_black_mask[pos[0]:pos[0]+h, pos[1]:pos[1]+w] = cropped_conn_mask[num]
                image_black[pos[0]:pos[0]+h, pos[1]:pos[1]+w] = small_image
                image_black_label[pos[0]:pos[0]+h, pos[1]:pos[1]+w] = cropped_full_NOT[num]
                prev_positions.append(pos)

        tmp_img = image_black.copy()

        image_black = np.where(image_black == [0, 0, 0], np.array([255, 255, 255], dtype=np.uint8), image_black)

        image_black = np.where(image_black == [255, 255, 255], background_photo, image_black)
        background_photo = cv2.add(background_photo, image_black)  
        dst = cv2.addWeighted(background_photo, 0.05, image_black, 0.95, 0.0)
        
        #blur edges
        blur = cv2.blur(dst,(5,5))
        mask = np.zeros(dst.shape, np.uint8)
        gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, (255,255,255),10) #changed from 8
        output = np.where(mask==np.array([255, 255, 255]), blur, dst) #output photo
        
    
        #make image with mask value    
        mask_NOT_copy = image_black_label.copy()
        mask_NOT_copy = cv2.cvtColor(mask_NOT_copy, cv2.COLOR_BGR2GRAY)
        out_photo_copy = output.copy()
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for random_contour in contours: 
            cv2.drawContours(out_photo_copy, [random_contour], -1, (0,100,0), 5) #draw contour
            x, y, w, h = cv2.boundingRect(random_contour) #boundingbox around the contour
            while True: #search point inside contour
                random_point = (np.random.randint(x, x + w), np.random.randint(y, y + h))
                if cv2.pointPolygonTest(random_contour, random_point, False) >= 0:
                    value = mask_NOT_copy[random_point[1], random_point[0]] #write value from the contour
                    out_photo_copy = cv2.putText(out_photo_copy, str(value-1), random_point, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 4, cv2.LINE_AA)     
                    break
        
        real_conn_num = len(contours)
        if (b < 100 and b >= 10):
            out_file_name = '0' + str(b) + '.png'
            out_test_file = out_folder_test + str(real_conn_num)+ '_' +str(conn_num) + '_' + '0' + str(b) + '.png'
        elif b < 10:
            out_file_name = '00' + str(b) + '.png'
            out_test_file = out_folder_test + str(real_conn_num)+ '_' +str(conn_num) + '_' + '00' + str(b) + '.png'
        else:
            out_file_name = str(b) + '.png'
            out_test_file = out_folder_test + str(real_conn_num)+ '_' +str(conn_num) + '_' + str(b) + '.png'

        if real_conn_num==conn_num:
            # resize image
            im_h = cv2.hconcat([image_black_mask, out_photo_copy])
            scale_percent = 40 # percent of original size
            width = int(im_h.shape[1] * scale_percent / 100)
            height = int(im_h.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(im_h, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(out_test_file, im_h)
                    
            save_photos(out_file_name, output, image_black_label, image_black_mask)
                
        
    print('DONE')
    
make_augumentations()
find_polygons_save_txt_file()
wandb.finish()
