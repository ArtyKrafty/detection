import os
import cv2

from detection.pipe import Pipeline


class SaveImg(Pipeline):

    """
    
    Класс, для сохранения изображений по указанному пути
    
    """


    def __init__(self, src, path, image_ext="jpg", jpg_quality=None, png_compression=None):
        self.src = src
        self.path = path
        self.image_ext = image_ext
        self.jpg_quality = jpg_quality
        super().__init__()
    def map(self, data):
        image = data[self.src]
        image_id = data["image_id"]
        output = image_id.split(os.path.sep)
        dirname = output[:-1]
        dirname = self.path
        filename = f"{output[-1].rsplit('.', 1)[0]}.{self.image_ext}"
        path = os.path.join(dirname, filename)

        if self.image_ext == "jpg":
            cv2.imwrite(path, image,
                        (cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality) if self.jpg_quality else None)
        elif self.image_ext == "png":
            cv2.imwrite(path, image,
                        (cv2.IMWRITE_PNG_COMPRESSION, self.png_compression) if self.png_compression else None)
        else:
            raise Exception("Unsupported image format")

        return data