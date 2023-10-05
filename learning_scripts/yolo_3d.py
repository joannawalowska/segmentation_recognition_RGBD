from ultralytics import YOLO

model = YOLO('/home/dataset/yolo_4chan.yaml')

model.train(data='/home/dataset/config_yolo_4chan.yaml', augment=False, batch=10, verbose=True, epochs=400, close_mosaic=0, imgsz=640, copy_paste=0.0, hsv_h = 0.0, hsv_v = 0.0, save = True, mosaic = 0.0, degrees = 0.0, patience=20)
