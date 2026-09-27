

from ultralytics import YOLO
from insightface.app import FaceAnalysis
import cv2


frame = cv2.imread('image.png')


model = YOLO('models/yolov8n.pt')

face_handler = FaceAnalysis(
    'antelopev2',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    root='.'
)
face_handler.prepare(ctx_id=0, det_size=(320, 320))


resuilt = model.predict(frame)[0]

for person in resuilt.boxes:
    x1, y1, x2, y2 = map(int, person.xyxy[0][::1])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (123, 23, 123), 1)
    persons=frame[y1:y2,x1:x2]
    face=face_handler.get(persons,10)
    fx1,fy1,fx2,fy2=map(int,face[0].bbox)
    



    # cv2.imshow('frame', persons)
    # cv2.waitKey(0)

