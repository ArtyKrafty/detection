
# https://github.com/jagin/detectron2-pipeline/blob/master/pipeline/separate_background.py

from detection.pipe import Pipeline
import numpy as np
import cv2


class Separate(Pipeline):

    """
    
    Класс, для отделения фона изображения.
    Предсказанная маска инвертируется, по сути мы выделяем
    обратное маске часть изображения и не наносим аннотацию 
    в виде классов и уверенности
    
    """

    def __init__(self, vis, me_kernel=(7, 7), bg_kernel=(21, 21), desaturate=True):
        self.vis = vis
        self.me_kernel = me_kernel
        self.bg_kernel = bg_kernel
        self.desaturate = desaturate

        super().__init__()

    def map(self, data):
        self.separate_background(data)

        return data

    def separate_background(self, data):
        if "predictions" not in data:
            return

        predictions = data["predictions"]
        if "instances" not in predictions:
            return

        instances = predictions["instances"]
        if not instances.has("pred_masks"):
            return
        mask = instances.pred_masks.cpu().sum(0) >= 1
        mask = mask.numpy().astype("uint8")*255
        mask = np.stack([mask, mask, mask], axis=2)
        mask = cv2.GaussianBlur(mask, self.me_kernel, 0)
        foreground = data["image"]
        background = cv2.GaussianBlur(foreground, self.bg_kernel, 0)

        if self.desaturate:

            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

        foreground = foreground.astype(float)
        background = background.astype(float)
        mask = mask.astype(float)/255.0
        foreground = cv2.multiply(foreground, mask)
        background = cv2.multiply(background, 1.0 - mask)
        dst_image = cv2.add(foreground, background)
        data[self.vis] = dst_image.astype("uint8")