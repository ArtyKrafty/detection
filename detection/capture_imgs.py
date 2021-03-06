import cv2

from detection.pipe import Pipeline
import detection.utils.files as fs



class CaptureImgs(Pipeline):

    """
    
    Класс входит в pipeline обработки изображений. 
    Принимает на вход путь к файлам. Собирает словарь,
    составленный из названия изображения в папке input,
    изображения и передает словарь в модель для предсказаний

    """

    def __init__(self, path, valid_exts=(".jpg", ".png", ".jpeg"), level=None, contains=None):
        self.path = path
        self.valid_exts = valid_exts
        self.level = level
        self.contains = contains

        super().__init__()

    def generator(self):

        source = fs.list_files(self.path, self.valid_exts, self.level, self.contains)
        while self.has_next():
            try:
                image_file = next(source)
                image = cv2.imread(image_file)
                
                data = {
                    "image_id": image_file,
                    "image": image
                }

                if self.filter(data):
                    yield self.map(data)
            except StopIteration:
                return