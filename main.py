import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv
from supervision.draw.color import ColorPalette
from supervision import Detections, BoxAnnotator

class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using  device:", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness = 3, text_thickness=3, text_scale=1.5)

    def load_model(self):
        model = YOLO("yolov8m.pt")  
        model.fuse()
        return model      

    def predict(self, frame):
        results = self.model(frame)
        return results  
    
    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
        
        for result in results[0]:
            class_id = result.boxes.cls.cpu.numpy().astype(int)

            if class_id == 0:
              xyxys.append(result.boxes.xyxy.cpu().numpy())
              confidences.append(result.boxes.conf.cpu().numpy())
              class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            
        
        detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
    
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections]
        
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        return frame
    
    
    
    
       

