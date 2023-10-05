

import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image
import cv2
import os
import numpy as np
from torchvision.transforms import ToTensor
import glob
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import zoom
from skimage.transform import rescale
from skimage.transform import resize
import torch
import torchvision
from torchvision.transforms import ToPILImage
# import wandb

# os.system('wandb login')
# a = wandb.init(
#     # set the wandb project where this run will be logged
#     project="augumentation"
# )

camera_matrix = [[606.6588134765625, 0, 638.6158447265625],
    [0, 606.464111328125, 365.907958984375],
    [0, 0, 1]]

Cx = camera_matrix[0][2]
Cy = camera_matrix[1][2]
Fx = camera_matrix[0][0]
Fy = camera_matrix[1][1]





path_background_png = 'background_png'
files_background_png = glob.glob(path_background_png + '/*')

path_background_tiff = 'background_tiff'
files_background_tiff = glob.glob(path_background_tiff + '/*')

path_mask = 'conn_mask'
files_mask = glob.glob(path_mask + '/*')

path_conn_tiff = 'conn_tiff'
files_image = glob.glob(path_conn_tiff + '/*')

path_conn_png = 'conn_png'
files_image = glob.glob(path_conn_png  + '/*')


def save_photos(name, name_tiff, end_photo, end_3d, end_mask, end_tiff):
    photo_name = 'end_augmented_photo/' + name
    mask_NOT_name = 'end_augmented_3d/' + name
    mask_name = 'end_augmented_mask/' + name
    tiff_name = 'end_tiff/' + name_tiff
    cv2.imwrite(photo_name, end_photo)
#     cv2.imwrite(mask_NOT_name, end_3d)
    cv2.imwrite(mask_name, end_mask) 
    end_3d.save(mask_NOT_name)
    cv2.imwrite(tiff_name, end_tiff)

def find_poligon():
        input_dir = 'end_augmented_mask/'         
        output_dir = './output_txt/'
        
        for j in os.listdir(input_dir):
            mask_value = []
            mask_polygon = []
            polygons = []
            image_path = os.path.join(input_dir, j)
            mask_1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask_2 = cv2.imread(image_path)

            _, mask = cv2.threshold(mask_1, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for random_contour in contours: #contours on actual photo
                cv2.drawContours(mask_2, [random_contour], -1, (0,100,0), 3) 
                x, y, w, h = cv2.boundingRect(random_contour) #boundingbox wokół konturu
                while True: 
                    random_point = (np.random.randint(x, x + w), np.random.randint(y, y + h))
                    if cv2.pointPolygonTest(random_contour, random_point, False) >= 0:
                        value = mask_2[random_point[1], random_point[0]][0] 
                        mask_value.append(value) 
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

def main():
    
    for b in range(600, 1000):
        back_number = random.randint(0, len(files_background_png)-1)
        background_png = Image.open(files_background_png[back_number])

        conn_num = random.randint(3, 7)

        file_path = files_background_png[back_number]
        file_name = os.path.split(file_path)[-1]
        file_name = os.path.splitext(file_name)[0]


        background_tiff = Image.open('background_tiff/'+f'{file_name}'+".tiff")
        conn_num = random.randint(2, 3)

        conn_png_list1 = []
        mask_list1 = []
        conn_tiff_list1 = []

        conn_png_array1 =[]
        conn_tiff_array1 = []
        mask_array1 = []

        a = True

        for i in range(conn_num):
            conn_folder = files_mask[random.randint(0, len(files_mask)-1)]
            if a:
                conn_mask_png = glob.glob(conn_folder + '/*.png')
                conn_mask = conn_mask_png[random.randint(0, len(conn_mask_png)-1)]
                conn_number = '/'.join(conn_mask.split('/')[1:-1]) 
                result1 = conn_mask.split('/')[-1].split('.png')[0]
                conn_mask_tiff = glob.glob(f'conn_tiff/{conn_number}/{result1}.tiff')
                conn_tiff = conn_mask_tiff[random.randint(0, len(conn_mask_tiff)-1)]

                result_png = '/' + '/'.join(conn_mask.split('/')[1:])

                conn_mask_png = glob.glob(f'conn_png{result_png}')
                conn_png = conn_mask_png[random.randint(0, len(conn_mask_png)-1)]
                conn_mask = Image.open(conn_mask)
                conn_mask = conn_mask.convert('L')
                
                conn_tiff = Image.open(conn_tiff)
                conn_png  = Image.open(conn_png)

                mask_array_1 = np.array(conn_mask)
                tiff_array = np.array(conn_tiff)
                png_array = np.array(conn_png)


                conn_tiff_list1.append(conn_tiff)
                conn_png_list1.append(conn_png)
                mask_list1.append(conn_mask)

                conn_png_array1.append(png_array)
                conn_tiff_array1.append(tiff_array)
                mask_array1.append(mask_array_1)

        resized_image_1 = []
        y_start_1 = []
        mniejsza_wysokosc_1 = []
        x_start_1 = []
        mniejsza_szerokosc_1 = []
        cropped_conn_png_2 = []
        mask_roi_2 = []
        x_2 = []
        y_2 = []
        mask_dilated_3 = []

        for subarray, sub_tiff, sub_png, maskk in zip(mask_array1, conn_tiff_array1, conn_png_array1, mask_list1):
                        mask_array = subarray
                        conn_tiff_array = sub_tiff
                        conn_png_array = sub_png
                        mask_png = maskk#[1]
                        mask_png_3 = maskk

                        indices = np.argwhere(mask_array != 0)

                        #szukanie dla środku
                        leftmost = np.min(indices[:, 1])
                        rightmost = np.max(indices[:, 1])
                        topmost = np.min(indices[:, 0])
                        bottommost = np.max(indices[:, 0])

                        #Srodki
                        Uo = int((leftmost + rightmost) / 2)
                        Vo = int((topmost + bottommost) / 2)


                        y_indices, x_indices = np.where(mask_array != 0 )

                        U_Left = np.min(x_indices)
                        U_Right = np.max(x_indices)
                        V_Top = np.min(y_indices)
                        V_Down = np.max(y_indices)

                        V_Left = y_indices[np.argmin(x_indices)]
                        V_Right = y_indices[np.argmax(x_indices)]
                        U_Top = x_indices[np.argmin(y_indices)]
                        U_Down = x_indices[np.argmax(y_indices)]

                        tiff_conn_array_supplement = conn_tiff_array.copy()
                        tiff_conn_array_supplement[tiff_conn_array_supplement == 0] = random.randint(int(np.min(tiff_conn_array_supplement[tiff_conn_array_supplement != 0])), int(np.max(tiff_conn_array_supplement)))

                        Zo = tiff_conn_array_supplement[Vo, Uo]
                        Z_Left = tiff_conn_array_supplement[V_Left, U_Left]
                        Z_Right = tiff_conn_array_supplement[V_Right, U_Right]
                        Z_Top = tiff_conn_array_supplement[V_Top, U_Top]
                        Z_Down = tiff_conn_array_supplement[V_Down, U_Down]

                        conn_tiff_with_mask = cv2.bitwise_and(tiff_conn_array_supplement , tiff_conn_array_supplement , mask=mask_array)

                        Xo = ((Uo - Cx)*Zo)/Fx
                        Yo = ((Vo - Cy)*Zo)/Fy

                        X_Left = ((U_Left - Cx)*Z_Left)/Fx
                        Y_Left = ((V_Left - Cy)*Z_Left)/Fy

                        X_Right = ((U_Right - Cx)*Z_Right)/Fx
                        Y_Right = ((V_Right - Cy)*Z_Right)/Fy

                        X_Top = ((U_Top - Cx)*Z_Top)/Fx
                        Y_Top = ((V_Top - Cy)*Z_Top)/Fy

                        X_Down = ((U_Down - Cx)*Z_Down)/Fx
                        Y_Down = ((V_Down - Cy)*Z_Down)/Fy

                        #wchodzimy w tło
                        tiff_back = background_tiff
                        back_array = np.array(background_tiff)
                        width, height = tiff_back.size


                        # Oblicz środek
                        Ut = random.randint(int(width*0.1), int(width*0.6))
                        Vt = random.randint(int(height*0.1), int(height*0.6))

                        Zt = back_array[Vt, Ut]
                        Xt = ((Ut - Cx)*Zt)/Fx
                        Yt = ((Vt - Cy)*Zt)/Fy

                        #tło dla wtyczkitu dodać 
                        Xtw = Xt
                        Ytw = Yt
                        Ztw = Zt - random.randint(50, 250)#-1000
                        Utw = ((Xtw*Fx)/Ztw)+Cx
                        Vtw = ((Ytw*Fy)/Ztw)+Cy

                        Xtw_Left = X_Left - Xo + Xtw
                        Ytw_Left = Y_Left - Yo + Ytw
                        Ztw_Left = Z_Left - Zo + Ztw
                        Utw_Left = ((Xtw_Left*Fx)/Ztw_Left) +Cx
                        Vtw_Left = ((Ytw_Left*Fy)/Ztw_Left) +Cy

                        Xtw_Right = X_Right - Xo + Xtw
                        Ytw_Right = Y_Right - Yo + Ytw
                        Ztw_Right = Z_Right - Zo + Ztw
                        Utw_Right = ((Xtw_Right*Fx)/Ztw_Right) +Cx
                        Vtw_Right = ((Ytw_Right*Fy)/Ztw_Right) +Cy

                        Xtw_Top = X_Top - Xo + Xtw
                        Ytw_Top = Y_Top - Yo + Ytw
                        Ztw_Top = Z_Top - Zo + Ztw
                        Utw_Top = ((Xtw_Top*Fx)/Ztw_Top) +Cx
                        Vtw_Top = ((Ytw_Top*Fy)/Ztw_Top) +Cy

                        Xtw_Down = X_Down - Xo + Xtw
                        Ytw_Down = Y_Down - Yo + Ytw
                        Ztw_Down = Z_Down - Zo + Ztw
                        Utw_Down = ((Xtw_Down*Fx)/Ztw_Down) +Cx
                        Vtw_Down = ((Ytw_Down*Fy)/Ztw_Down) +Cy


                        new_width =abs(int(Utw_Right-Utw_Left))
                        new_high = abs(int(Vtw_Down-Vtw_Top))

                        old_high = abs(int(V_Down - V_Top))
                        old_width = abs(int(U_Right - U_Left))

                        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        x, y, w, h = cv2.boundingRect(contours[0])

                        bbox_image = cv2.rectangle(conn_tiff_with_mask, (x, y), (x + w, y + h), (0, 0, 0), 0)
                        cropped_conn = bbox_image[y:y+h, x:x+w]

                        cropped_conn_array = np.array(cropped_conn)
                        resized_image = rescale(cropped_conn_array, (new_high / cropped_conn_array.shape[0], new_width / cropped_conn_array.shape[1]), preserve_range=True)

                        resized_image = Image.fromarray(resized_image.astype(np.uint8))

                        conn_png_with_mask = cv2.bitwise_and(conn_png_array , conn_png_array , mask=mask_array)

                        bbox_conn_png = cv2.rectangle(conn_png_with_mask, (x, y), (x + w, y + h), (0, 0, 0), 0)
                        cropped_conn_png = bbox_conn_png [y:y+h, x:x+w]

                        #tu mniejsze zdjęcie to wyższe
                        resized_iconn_png = resize(cropped_conn_png, (new_high, new_width))
                        resized_iconn_array = np.array(resized_iconn_png)

                        # Tworzenie maski czarno-białej
                        mm = mask_array.copy()
                        bbox_conn_mask_png = cv2.rectangle(mm, (x, y), (x + w, y + h), (0, 0, 0), 0)
                        cropped_conn_mask_png = bbox_conn_mask_png [y:y+h, x:x+w]
                        resized_conn_mask_png = resize(cropped_conn_mask_png, (new_high, new_width))

                        x_start = int(Utw_Left)
                        y_start = int(Vtw_Top)

                        back_png_array = np.array(background_png)

                        resized_image = np.array(resized_image)
                        back_array_to_tiff = back_array.copy()
                        back_array_to_png = back_png_array.copy()

                        mniejsza_szerokosc = resized_image.shape[1]
                        mniejsza_wysokosc = resized_image.shape[0]

                        resized_image_1.append(resized_image)
                        y_start_1.append(y_start)
                        mniejsza_wysokosc_1.append(mniejsza_wysokosc)
                        x_start_1.append(x_start)
                        mniejsza_szerokosc_1.append(mniejsza_szerokosc)

                        width = conn_png_array.shape[1]
                        high = conn_png_array.shape[0]
                        scale_high = int(high*(new_high/old_high))
                        scale_width = int(width*(new_width/old_width))

                        resized_conn_png = (resize(conn_png_array, (scale_high, scale_width)) * 255).astype(np.uint8)
                        resized_conn_mask_3 = resize(mask_array, (scale_high, scale_width), preserve_range=True).astype(np.uint8)
                        resized_conn_mask_png = resize(mask_array, (scale_high, scale_width), preserve_range=True).astype(np.uint8)

                        mask_dilated = resized_conn_mask_png

                        # Kontury
                        threshold = 0
                        binary_mask = (mask_dilated > threshold).astype(np.uint8)
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        mask_dilated_int32 = np.int32(mask_dilated)

                        # Wybór największego konturu (obiektu)
                        contour = max(contours, key=cv2.contourArea)

                        # Znalezienie bounding boxa dla konturu
                        x, y, w, h = cv2.boundingRect(contour)

                        cropped_conn_png = resized_conn_png[y:y+h, x:x+w]
                        cropped_conn_mask_3 = resized_conn_mask_3[y:y+h, x:x+w]
                        cropped_conn_mask = resized_conn_mask_png[y:y+h, x:x+w]

                        width = cropped_conn_png.shape[1]
                        high = cropped_conn_png.shape[0]

                        cropped_conn_mask[cropped_conn_mask <= 0] = 0
                        cropped_conn_mask[cropped_conn_mask > 0] = 1

                        unikalne_wartosci = np.unique(resized_conn_mask_png)

                        background = background_png.copy()
                        background_array = np.array(background)

                        kernel = np.ones((5, 5), np.uint8)  
                        cropped_conn_mask_3[cropped_conn_mask_3 != 0] = np.max(cropped_conn_mask_3)

                        mask_dilated_3.append(cropped_conn_mask_3)
                        mask_roi = cropped_conn_mask_3 > 0

                        output_array = background_array.copy().astype(np.uint8)

                        y = int(Vtw_Top)
                        x = int(Utw_Left)

                        cropped_conn_png_2.append(cropped_conn_png)
                        mask_roi_2.append(mask_roi)
                        
                        x_2.append(x)
                        y_2.append(y)

        back_array_to_tiff_2 = np.array(background_tiff)
        back_array_to_tiff = back_array_to_tiff_2.copy()

        for resized_image_1_1, y_start_1_1, mniejsza_wysokosc_1_1, x_start_1_1, mniejsza_szerokosc_1_1 in zip(resized_image_1, y_start_1, mniejsza_wysokosc_1, x_start_1, mniejsza_szerokosc_1):
                            back_array_to_tiff= back_array_to_tiff

                            resized_image_for = resized_image_1_1
                            y_start_for = y_start_1_1
                            x_start_for = x_start_1_1
                            mniejsza_wysokosc_for = mniejsza_wysokosc_1_1
                            mniejsza_szerokosc_for = mniejsza_szerokosc_1_1

                            ## tworzenie maski warunku, gdzie resized_image jest różne od 0
                            warunek = resized_image_for != 0

                            # okleśla współrzędne do przypisania wartości
                            rows, cols = np.where(warunek)
                            height_for, width_for = resized_image_for.shape

                            # przypisuje wartości z resized_image do back_array_to_tiff na podstawie współrzędnych
                            for row in range(height_for):
                                for col in range(width_for):
                                    y_coord = y_start_for + row
                                    x_coord = x_start_for + col
                                    if 0 <= y_coord < back_array_to_tiff.shape[0] and 0 <= x_coord < back_array_to_tiff.shape[1]:
                                        back_array_to_tiff[y_coord, x_coord] = resized_image_for[row, col]


        output_array1 = np.array(background_png)
        output_array = output_array1.copy()


        for cropped_conn_png_2_2, mask_roi_2_2, x_2_2, y_2_2 in zip(cropped_conn_png_2, mask_roi_2, x_2, y_2):

                        output_array = output_array
                        cropped_conn_png_for = cropped_conn_png_2_2
                        mask_roi_for = mask_roi_2_2
                        erosion_size = 7              
                        kernel = np.ones((erosion_size, erosion_size), np.uint8)

                        # Wykonaj erozję na masce
                        mask_roi_for = cv2.erode(mask_roi_for.astype(np.uint8), kernel, iterations=1)

                        x_for = x_2_2
                        y_for = y_2_2
 
                        width = cropped_conn_png_for.shape[1]
                        high = cropped_conn_png_for.shape[0]

                        for i in range(high):
                            for j in range(width):
                                if mask_roi_for[i, j]:
                                    y_coord = y_for + i
                                    x_coord = x_for + j
                                    if 0 <= y_coord < output_array.shape[0] and 0 <= x_coord < output_array.shape[1]:
                                        output_array[y_coord, x_coord] = cropped_conn_png_for[i, j]

                        output_array = output_array.astype(np.uint8)
                        output_image = Image.fromarray(output_array)

        output_array3 = np.array(background_png)
        output_array3 = output_array3.copy()

        output_array3 = np.copy(output_array3)
        output_array3 = np.zeros_like(output_array3)


        for cropped_conn_png_3_3, x_3_3, y_3_3,  mask_roi_3_3 in zip(mask_dilated_3, x_2, y_2,  mask_roi_2):
                        output_array3 = output_array3
                        szerokosc = output_array3.shape[1]
                        wysokosc = output_array3.shape[0]
                        
                        przeciwnosc_maski = mask_roi_3_3
                        cropped_conn_png_for3 = cropped_conn_png_3_3
                        mask_roi_for3 = mask_roi_3_3

                        x_for3 = x_3_3
                        y_for3 = y_3_3

                        width = cropped_conn_png_for3.shape[1]
                        high = cropped_conn_png_for3.shape[0]

                        for i in range(high):
                            for j in range(width):
                                if przeciwnosc_maski[i, j]:
                                    y_coord = y_for3 + i
                                    x_coord = x_for3 + j
                                    if 0 <= y_coord < output_array3.shape[0] and 0 <= x_coord < output_array3.shape[1]:
                                        output_array3[y_coord, x_coord] = cropped_conn_png_for3[i, j]
   
                        output_array3 = output_array3.astype(np.uint8)
                        output_image3 = Image.fromarray(output_array3)

        tiff = back_array_to_tiff.copy()
        tiff = ToTensor()(tiff)

        jpg = output_array.copy()
        jpg = ToTensor()(jpg)

        c = torch.cat((tiff, jpg), 0)
        to_pil = ToPILImage()
        c = to_pil(c)

        if (b < 100 and b >= 10):
                out_file_name = '0' + str(b)  +'.png'
                out_file_tiff = '0' + str(b)  +'.tiff'
        elif b < 10:
                out_file_name = '00' + str(b) + '.png'
                out_file_tiff = '00' + str(b) + '.tiff'

        else:
                out_file_name = str(b) + '.png'
                out_file_tiff = str(b) + '.tiff'

        output_image = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)

        save_photos(out_file_name, out_file_tiff , output_image, c, output_array3, back_array_to_tiff)
        print("done")

main()
find_poligon()
# wandb.finish()
# print("done")
   



