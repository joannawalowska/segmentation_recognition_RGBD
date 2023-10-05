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

dic = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0), 5: (255, 0, 255)}

dic_names = {1: 'conn_a', 2: 'root', 3: 'box', 4: 'bag_a', 5: 'bag_b'}

class InterferenceUnet(Node):
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
        self.img_dep = self.bridge.imgmsg_to_cv2(dep, desired_encoding="32FC1")

        #PREDICT
        img_resized = cv2.resize(self.img_rgb, (768, 512), interpolation = cv2.INTER_NEAREST)
        img_tensor = self.conv_tens(np.array(img_resized, dtype=np.float32) /255)
        img_device = img_tensor.to(device='cuda')
        img_to_model = img_device.unsqueeze(0)
        
        pred = self.model(img_to_model)
        tmp = pred[0]
        softmax = nn.Softmax(dim=1)
        tmp = torch.argmax(softmax(tmp),axis=1)
        tmp = tmp.squeeze(0)
        prediction = np.array(tmp.cpu())


        pred_mask = prediction[:,:, np.newaxis]
        pred_mask = np.uint8(pred_mask)
        copy_image = img_resized.copy()
        copy_pred = cv2.merge((pred_mask,pred_mask,pred_mask))

        cp = np.where(copy_pred==1, (255, 0, 0), copy_pred)
        cp = np.where(cp==2, (0, 255, 0), cp)
        cp = np.where(cp==3, (0, 0, 255), cp)
        cp = np.where(cp==4, (255, 255, 0), cp)
        cp = np.where(cp==5, (255, 0, 255), cp)
      
        cp = cp.astype(np.uint8)

        with_labels = cv2.add(cp, copy_image)  
        dst = cv2.addWeighted(with_labels, 0.7, copy_image, 0.3, 0.0)
        
        contour_list = []
        #Get all contours
        mask_gray = cv2.cvtColor(copy_pred, cv2.COLOR_BGR2GRAY)
        for b in range(1,6):
          tmp = np.where(mask_gray==b, b, 0).astype(np.uint8)
          _, mask = cv2.threshold(tmp, 1, 255, cv2.THRESH_BINARY)
          contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          for i in contours:
            contour_list.append(i)
            
        for c in contour_list:
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),1)
            counter = 0
            while True and counter<500: 
                random_point = (np.random.randint(x, x + w), np.random.randint(y, y + h))
                counter += 1
                if cv2.pointPolygonTest(c, random_point, False) > 0:
                    value = pred_mask[random_point[1], random_point[0]][0] 
                    if value>0:
                        cv2.putText(dst, dic_names[int(value)], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        break 



        #PUBLISH
        msg = Image()
        msg = self.bridge.cv2_to_imgmsg(dst, encoding="rgb8")
        self.publisher_.publish(msg)


def main(args=None):
    model = torch.load('/home/chaneu/remodel/unet_model.pt')
    rclpy.init(args=args)

    subsc = InterferenceUnet(model)

    rclpy.spin(subsc)

    subsc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
