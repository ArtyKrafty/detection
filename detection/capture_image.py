import cv2

from detection.pipeline import Pipeline


class CaptureImage(Pipeline):

    def __init__(self, src):
        self.src = src

        super().__init__()

    def generator(self):
        image = cv2.imread(self.src)

        data = {
            "image_id": self.src,
            "image": image
        }

        if self.filter(data):
            yield self.map(data)