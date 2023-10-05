import rclpy
from rclpy.node import Node
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms
from ultralytics import YOLO
from torchvision.transforms import ToTensor
from PIL import Image as IMAGE
from torchvision.transforms import ToPILImage


dic = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 
           5: (0, 255, 255)}
dic_names = {0: 'conn_a', 1: 'root', 2: 'box', 3: 'bag_a', 4: 'bag_b'}

class InterferenceYolo(Node):
    def __init__(self, model):
        super().__init__('minimal_subscriber')
        self.bridge = CvBridge()
        self.img_rgb = None
        self.img_dep = None
        self.model = model
        self.conv_tens = transforms.ToTensor()
        

        #SUBSCRIBERS
        self.subscription_rgb = self.create_subscription(
            Image,
            '/rgb/image_raw', self.rgb_callback, 10)

        self.subscription_dep = self.create_subscription(
            Image,
            '/depth_to_rgb/image_raw',  self.dep_callback, 10)

        self.subscription_rgb_ = Subscriber(self, Image, '/rgb/image_raw')
        self.subscription_dep_ = Subscriber(self, Image, '/depth_to_rgb/image_raw')

        self.ts = ApproximateTimeSynchronizer([self.subscription_rgb_, self.subscription_dep_], 10, 0.1)
        self.ts.registerCallback(self.callback)

        #PUBLISHERS
        self.publisher_ = self.create_publisher(Image, '/output', 10)
 


    def dep_callback(self, msg):
        pass

    def rgb_callback(self, msg):

        pass

    def callback(self, rgb, dep):
        self.img_rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding="rgb8")
        self.img_rgb = cv2.resize(self.img_rgb, (1280, 720), interpolation = cv2.INTER_NEAREST)

        self.img_dep = self.bridge.imgmsg_to_cv2(dep, desired_encoding="32FC1")
        self.img_dep = cv2.resize(self.img_dep, (1280, 720), interpolation = cv2.INTER_NEAREST)


        #PREDICT
            #Prepare RGBD image
        tiff_tensor = ToTensor()(self.img_dep)
        png_tensor = ToTensor()(self.img_rgb)
        img_to_model2 = torch.cat((tiff_tensor, png_tensor), dim=0)
        img_to_model2 = ToPILImage()(img_to_model2)
        img_to_model2.save('dff.png')
        img_to_model = cv2.imread('dff.png', cv2.IMREAD_UNCHANGED)

        (r, d, b, g) = cv2.split(img_to_model)
        img_to_model = cv2.merge([r, g, b, d])
        pred = self.model(img_to_model)
        H, W, _ = self.img_rgb.shape
        dst = self.img_rgb
        
        img_copy = self.img_rgb
        copy_image = self.img_rgb
        pred_mask = np.zeros_like(copy_image)

        for se in pred:
            if se:
                clist = se.boxes.cls
                clss = []
                masks = []
                clean = np.zeros_like(img_copy)
                clean_bouding_box = np.zeros_like(img_copy)
                contour_list = []
                for cno in clist:
                    clss.append(int(cno))

                for j, mask in enumerate(se.masks.data):
                    mask = mask.cpu().numpy() * 255
                    mask = cv2.resize(mask,(W,H))
                    value = dic[clss[j]]

                    end_mask = cv2.merge((mask,mask,mask)) #3 channel mask
                    out_1 = (np.where(end_mask>0, value, end_mask)).astype(np.uint8)
                    copy_image = np.where(out_1>0, out_1, copy_image)
                    pred_mask = np.where(pred_mask == 0, end_mask, pred_mask)

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



        #PUBLISH
        msg = Image()
        msg = self.bridge.cv2_to_imgmsg(dst, encoding="rgb8")
        self.publisher_.publish(msg)
      

def main(args=None):
    model = YOLO("/home/chaneu/remodel/yolo_model_3d.pt")
    rclpy.init(args=args)

    subsc = InterferenceYolo(model)

    rclpy.spin(subsc)

    subsc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
