import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detection.pipe import Pipeline


class AnnotateImg(Pipeline):

    """
    
    Класс входит в pipeline обработки изображений. 
    Принимает на вход предсказания. Производит аннотацию изображения
    и передает итог в класс сохранения. Работает, 
    если не выбран параметр - отделение фона


    """

    def __init__(self, vis, metadata_name):
        self.vis = vis
        self.metadata_name = metadata_name
        self.cpu_device = torch.device("cpu")

        super().__init__()

    def map(self, data):
        dst_image = data["image"].copy()
        data[self.vis] = dst_image
        self.annotate_predictions(data)
        return data

    def annotate_predictions(self, data):
        if "predictions" not in data:
            return

        predictions = data["predictions"]
        dst_image = data[self.vis]
        dst_image = dst_image[..., ::-1]

        visualizer = Visualizer(dst_image, self.metadata_name)

        if "panoptic_seg" in predictions:
            pan_seg, seg_info = predictions["panoptic_seg"]
            vis_image = visualizer.draw_panoptic_seg_predictions(pan_seg.to(self.cpu_device),
                                                                 seg_info)
        elif "instances" in predictions:
            instances = predictions["instances"]
            vis_image = visualizer.draw_instance_predictions(instances.to(self.cpu_device))
        
        vis_image = cv2.cvtColor(vis_image.get_image(), cv2.COLOR_RGB2BGR)
        data[self.vis] = vis_image     
        
        