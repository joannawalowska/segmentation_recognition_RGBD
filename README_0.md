# Segmenatation and recognition objects using RGBD camera
The student's master's thesis includes research into the segmentation and object recognition capabilities of an automotive beam using an RGBD camera. In the course of the work, data was collected, labeled accordingly, creating annotations. Then the dataset was augmented using copy-paste augmentation. The next step was to teach YOLOv8 and Unet on RGB and RGBD data. The learned models were then tested using prepared inference scripts. Among the conclusions is that in this case depth did not improve the prediction results. One reason may be the insufficient dataset made available for testing, which does not allow the models to find all the relationships.
The picture below-placed is an example of augmentation RGBD. On the left RGB image, on the right depth.

![image](https://github.com/joannawalowska/segmentation_recognition_RGBD/assets/147088977/2a14ebd4-9dbb-4c53-b83c-58012668a88f)

Example outputs:
![image](https://github.com/joannawalowska/segmentation_recognition_RGBD/assets/147088977/6eb3351b-acaf-4b11-97cf-4d7dcd9311b0)

![image](https://github.com/joannawalowska/segmentation_recognition_RGBD/assets/147088977/11764fd4-6d71-4410-b003-064848758681)


## Functionalities delivered
- RGB data augmentation 
- RGBD data augmentation as new method segmentation for RGBD data
- teaching YOLOv8 on RGB data
- teaching YOLOv8 on RGBD data
- Unet teaching on RGB data
- Unet learning on depth data
- inference for single images 
- real-time model inference

## Requrements
- Robot Operating System 2 Humble
- Python 3.10
- albumentations
- numpy
- opencv-python
- Pillow
- setuptools 58.2.0
- scikit-learn 0.24.2
- timm
- torch 2.0.1
- torchvision 0.15.2
- tqdm
- torchmetrics
- ultralytics 8.0.117
- pytorch-lightning
- wandb

## Authors
- Joanna Wałowska
- Zuzanna Chałupka
