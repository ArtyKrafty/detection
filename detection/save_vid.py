import cv2
import os

from detection.pipe import Pipeline


class SaveVid(Pipeline):

    """
    
    Класс, для сохранения видео по указанному пути
    
    """

    def __init__(self, src, filename, fps=30, fourcc=('m', 'p', '4', 'v')):
        dirname = os.path.dirname(os.path.abspath(filename))
        os.makedirs(dirname, exist_ok=True)

        self.src = src
        self.filename = filename
        self.fps = fps
        self.writer = None
        self.fourcc = fourcc

        super().__init__()

    def map(self, data):
        image = data[self.src]

        if self.writer is None:
            h, w = image.shape[:2]
            self.writer = cv2.VideoWriter(
                filename=self.filename,
                fourcc=cv2.VideoWriter_fourcc(*self.fourcc),
                fps=self.fps,
                frameSize=(w, h),
                isColor=(image.ndim == 3))

        self.writer.write(image)

        return data

    def cleanup(self):
        if self.writer:
            self.writer.release()