import cv2

from detection.pipe import Pipeline


class DisplayVid(Pipeline):

    """
    
    Класс входит в pipeline обработки видео. 
    Принимает на вход путь к готовым файлам. Позволяет в реальном
    времени налюдать за процессом обработки. Необходимо Gpu
    
    """

    def __init__(self, src, window_name=None, org=None):
        self.src = src
        self.window_name = window_name if window_name else src

        cv2.startWindowThread()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        if org:
            x, y = org
            cv2.moveWindow(self.window_name, x, y)

        super().__init__()

    def map(self, data):
        image = data[self.src]

        cv2.imshow(self.window_name, image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            raise StopIteration
        return data

    def cleanup(self):
        cv2.destroyWindow(self.window_name)