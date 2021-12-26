from detection.pipeline import Pipeline
from detectron2.engine.defaults import DefaultPredictor


class Predict(Pipeline):

    def __init__(self, cfg):
        self.predictor = DefaultPredictor(cfg)
        
        super().__init__()

    def map(self, data):
        image = data["image"]
        predictions = self.predictor(image)
        data["predictions"] = predictions

        return data