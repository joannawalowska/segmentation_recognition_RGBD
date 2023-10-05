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
        self.photo_number = 0
        

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
        # timer_period = 0.1  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)


    def dep_callback(self, msg):
        pass

    def rgb_callback(self, msg):
        # print('rgb callback')
        pass

    def callback(self, rgb, dep):
        self.img_rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding="rgb8")
        self.img_dep = self.bridge.imgmsg_to_cv2(dep, desired_encoding="32FC1")
        print("before model")

        #PREDICT
        img_to_model = self.img_rgb
        print('predict')
        pred = self.model(img_to_model)
        print('after predict')
        H, W, _ = img_to_model.shape
        dst = self.img_rgb
        print(H, W)
        
        img_copy = img_to_model.copy()

        for se in pred:
            if se:
                clist = se.boxes.cls
                clss = []
                masks = []
                clean = np.zeros_like(img_to_model)
                clean_bouding_box = np.zeros_like(img_to_model)
                for cno in clist:
                    clss.append(int(cno))

                for j, mask in enumerate(se.masks.data):
                    mask = mask.cpu().numpy() * 255
                    mask = cv2.resize(mask,(W,H))
                    value = dic[clss[j]]

                    mask = mask[:,:, np.newaxis]
                    end_mask = cv2.merge((mask,mask,mask)) #3 channel mask
                    out_1 = (np.where(end_mask>0, value, end_mask)).astype(np.uint8)
                    img_to_model = np.where(out_1>0, out_1, img_to_model)

                    #serching contour to draw boudingboxes
                    _, mask0 = cv2.threshold(mask.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(mask0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    max_contour = max(contours, key = cv2.contourArea)
                    rect = cv2.boundingRect(max_contour)
                    x,y,w,h = rect

                    cv2.rectangle(img_to_model,(x,y),(x+w,y+h),(0,255,0),1)
                    cv2.putText(img_to_model, dic_names[clss[j]], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


                dst = cv2.addWeighted(img_to_model, 0.7, img_copy, 0.3, 0.0)




        self.photo_number += 1
        #PUBLISH
        msg = Image()
        msg = self.bridge.cv2_to_imgmsg(dst, encoding="rgb8")
        self.publisher_.publish(msg)
      

def main(args=None):
    model = YOLO('/home/chaneu/remodel/yolo_model.pt')
    rclpy.init(args=args)

    subsc = InterferenceYolo(model)

    rclpy.spin(subsc)

    subsc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
