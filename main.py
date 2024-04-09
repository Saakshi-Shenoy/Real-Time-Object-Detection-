import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

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

    
    

       

