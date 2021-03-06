import cv2

from detection.pipe import Pipeline

class CaptureVid(Pipeline):

    """
    
    Класс входит в pipeline обработки видео. 
    Принимает на вход путь к файлам. Собирает словарь,
    составленный из номера кадра, названия файла и изображения.
    Видео файл делится на фреймы, которые отдельно передаются 
    как изображения в модель, а затем собираются в видео вновь

    """

    def __init__(self, src):

        self.cap = cv2.VideoCapture(src)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames_per_second = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f'Количество кадров {self.frame_count}')
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.num_frames = 0

        super().__init__()

    def generator(self):
        while True:
            _, image = self.cap.read()
            if not _:
                break
            data = {
                "frame_num": self.num_frames,
                "image_id": f"{self.num_frames:04d}",
                "image": image,
            }
            self.num_frames += 1
            if self.filter(data):
                yield self.map(data)
            if self.num_frames > self.frame_count:
                break

    def cleanup(self):
        self.cap.stop()