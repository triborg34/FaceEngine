import threading
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import multiprocessing
import time

from camera import FreshestFrame

class CCTVMONITOR:
    def __init__(self, source):
        self.source = source  # Fixed typo: soruce -> source
        
        # Load model and move to CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT",device='cuda')
        self.model.to(self.device)  # Move model to device
        self.model.eval()
        
        self.coco_classes = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
        
        # For controlling detection frequency
        self.frame_count = 0
        self.detection_interval = 25
        
        # Shared variables for detection results (with locks for thread safety)
        self.detection_lock = threading.Lock()
        self.detection_results = None

    def detection(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened(): 
            print("Error: Could not open video source")
            return
            
        fresh = FreshestFrame(cap)
        cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cam", 960, 540)
        
        while fresh.is_alive():
            ret, frame = fresh.read()
            if not ret:
                continue
                
            self.frame_count += 1
            
            # Run detection every 25 frames
            if self.frame_count % self.detection_interval == 0:
                frame_copy = frame.copy()
                threading.Thread(target=self.detect, args=[frame_copy], daemon=True).start()
            
            # Draw detection results on current frame
            display_frame = self.draw_detections(frame)
            
            cv2.imshow('Cam', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        cap.release()
        fresh.release()
        cv2.destroyAllWindows()

    def detect(self, frame):
        try:
            # Convert frame to tensor and move to device
            frame_tensor = F.to_tensor(frame).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(frame_tensor)

            boxes = predictions[0]["boxes"].cpu().numpy()  # Move back to CPU for processing
            labels = predictions[0]["labels"].cpu().numpy()
            scores = predictions[0]["scores"].cpu().numpy()
            
            # Store results in shared variable with thread lock
            with self.detection_lock:
                self.detection_results = {
                    'boxes': boxes,
                    'labels': labels,
                    'scores': scores,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            print(f"Detection error: {e}")

    def draw_detections(self, frame):
        # Get current detection results
        with self.detection_lock:
            if self.detection_results is None:
                return frame
            
            # Use results if they're recent (within 2 seconds)
            if time.time() - self.detection_results['timestamp'] > 2.0:
                return frame
                
            boxes = self.detection_results['boxes']
            labels = self.detection_results['labels']
            scores = self.detection_results['scores']
        
        # Draw detections on frame
        display_frame = frame.copy()
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.6:  # confidence threshold
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{self.coco_classes[label]}: {score:.2f}"
                cv2.putText(display_frame, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return display_frame


if __name__ == "__main__":
    cv2.setNumThreads(multiprocessing.cpu_count())
    monitor = CCTVMONITOR('rtsp://192.168.1.8:554/stream')
    monitor.detection()