import requests
import os
import io
from tqdm import tqdm
from PIL import Image
from detection.capture_web import CaptureWeb
from detection.predict import Predict
from detection.save_web import SaveWeb
from detection.annotate_img import AnnotateImg
from detection.separate_background import Separate
from detection.utils import detectron
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from flask import Flask, render_template, request, send_file


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/home", methods=['POST', 'GET'])
def main():

    method = request.method
    values = [x for x in request.form.values()]
    confidence_threshold = values[0]
    mode = values[1]

    if method == 'GET':
        src = request.args.get("url")
    elif method == 'POST':
        src = Image.open(request.files['file'].stream)

    capture_images = CaptureWeb(src)

    cfg = detectron.setup_cfg(config_file="./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                              weights_file=None,
                              confidence_threshold=confidence_threshold)
    predict = Predict(cfg)

    if mode == "separate_background":
        separate = Separate("vis_image")
        annotate_image = None
    else:
        separate = None
        metadata_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        metadata_name = MetadataCatalog.get(metadata_name)
        annotate_image = AnnotateImg("vis_image", metadata_name)

    save_image = SaveWeb("vis_image")

    pipeline = (capture_images |
              predict |
              separate |
              annotate_image |
              save_image)



if __name__ == "__main__":

    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)