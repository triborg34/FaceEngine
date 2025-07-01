
# import cv2
# import urllib.request

# import numpy as np
# d =urllib.request.urlretrieve('http://127.0.0.1:8090/api/files/collection/xd1mivsg192584q/s4zmi7yhrex_99mlpz1l83.unknown_1.jpg', "uploads/local-filename.jpg")

# print(d[0])
from ultralytics import YOLO
import cv2

model=YOLO('models/yolov8n.pt')
frame=cv2.imread(r'C:\Users\Microsoft\Downloads\aref.jpg')
frame=cv2.resize(frame,(640,640))

resuilt=model.predict(frame)[0]

x1,y1,x2,y2=map(int,resuilt.boxes.xyxy[0][:4])

hc=frame[y1:y2,x1:x2]

cv2.imshow('mat',hc)
cv2.waitKey(0)