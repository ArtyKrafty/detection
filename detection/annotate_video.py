import cv2
import torch
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detection.pipeline import Pipeline

class AnnotateVideo(Pipeline):
    def __init__(self, dst, metadata, instance_mode=ColorMode.IMAGE):
        self.dst = dst
        self.metadata = metadata
        self.instance_mode = instance_mode
        self.cpu_device = torch.device("cpu")
        self.video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)
        

        super().__init__()

    def map(self, data):
        dst_image = data["image"].copy()
        data[self.dst] = dst_image
        self.annotate_predictions(data)
        return data

    def annotate_predictions(self, data):
        if "predictions" not in data:
            return
        dst_image = data[self.dst]
        outputs = data["predictions"]    
        visualizer = Visualizer(data["image"][..., ::-1], self.metadata)
        vis_image = visualizer.draw_instance_predictions(outputs["instances"].to(self.cpu_device))
        # vis_image = self.video_visualizer.draw_instance_predictions(data["image"][..., ::-1], outputs["instances"].to(self.cpu_device))
        vis_image = cv2.cvtColor(vis_image.get_image(), cv2.COLOR_RGB2BGR)
        data[self.dst] = vis_image
