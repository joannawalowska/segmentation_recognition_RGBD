#!/usr/bin/env python
# coding: utf-8

from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt') 

model.train(data='/home/dataset/config_yolo.yaml', epochs=400, imgsz=640, batch=10 ,copy_paste=0.8, hsv_h = 0.0, hsv_v = 0.0, save = True, mosaic = 0.8, degrees = 0.5, patience=20)



