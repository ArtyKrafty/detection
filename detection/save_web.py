import os
import cv2

from detection.pipe import Pipeline


class SaveWeb(Pipeline):

    def __init__(self, src, path, image_ext="jpg"):
        self.src = src
        self.image_ext = image_ext
        self.jpg_quality = 95
        self.png_compression = 3

        super().__init__()

    def map(self, data):
        
        image = data[self.src]
        image_id = data["image_id"]
        io_obj = io.BytesIO()
        proccesed_img.save(file_object, 'jpg')
        io_obj.seek(0)
        
        data = send_file(file_object, mimetype='image/jpeg')

        return data