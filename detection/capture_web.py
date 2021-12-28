import cv2
from skimage import io
from PIL import Image
from detection.pipe import Pipeline


class CaptureWeb(Pipeline):

    def __init__(self, src):
        self.src = src
        self.method = method

        super().__init__()

    
    def generator(self):

        image = io.imread(self.src)
        image = cv2.imread(image)



        data = {
            "image_id": self.src,
            "image": image
        }

        if self.filter(data):
            yield self.map(data)
