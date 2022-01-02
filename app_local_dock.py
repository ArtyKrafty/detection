import detectron2
print(f"Detectron2 version is {detectron2.__version__}")
import requests
import os
import cv2
import numpy as np
import io
import torch, torchvision
from PIL import Image
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from flask import Flask, render_template, request, send_file
from detectron2.engine.defaults import DefaultPredictor
from detection.utils import detectron
from detectron2 import model_zoo

# скрипт app_local_dock.py нужен для работы web-приложения на локальной или виртуальной машине через docker

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
app = Flask(__name__)
UPLOAD_FOLDER = os.environ['UPLOADS']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST', 'GET'])
def main():
    app.logger.debug("PATH: %s", os.path.join(app.config['UPLOAD_FOLDER'], "image"))
    
     
    method = request.method
    app.logger.debug("Method: %s", method)
    if method == 'GET':
        url = request.args.get("url")
        app.logger.debug("URL: %s", url)
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        app.logger.debug("image: %s", image)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], "image_001.jpeg"))
        src = os.path.join(app.config['UPLOAD_FOLDER'], "image_001.jpeg")
        app.logger.debug("src: %s", src)
        mode = "instance_segmentation"
        CONFIDENCE = .7
        app.logger.debug("CONFIDENCE: %s", CONFIDENCE)
    elif method == 'POST':
        try:
            file = request.files['file']
            app.logger.debug("file: %s", file)
            if file:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], "image_001.jpeg"))
                src = os.path.join(app.config['UPLOAD_FOLDER'], "image_001.jpeg")
                app.logger.debug("src: %s", src)
                mode = request.form["mode"]
                CONFIDENCE = float(request.form["confidence"])
                app.logger.debug("CONFIDENCE: %s", CONFIDENCE)

        except:
            return render_template("error.html")

    image = cv2.imread(src)
    app.logger.debug("image: %s", image.shape)
    if mode in ['separate_background', 'instance_segmentation']:
    
        app.logger.debug("mode: %s", mode)
        cfg = detectron.setup_cfg(config_file=model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"),
                                  weights_file=model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"),
                                  confidence_threshold=CONFIDENCE)
      
        predictor = DefaultPredictor(cfg)
        app.logger.debug("cfg: %s", predictor) 
        outputs = predictor(image)
        if mode == 'instance_segmentation':
            metadata_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
            metadata_name = MetadataCatalog.get(metadata_name)
            visualizer = Visualizer(image[...,::-1], metadata_name)
            vis_image = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
            vis_image = Image.fromarray(np.uint8(vis_image.get_image()))

        else:
            me_kernel=(7, 7)
            bg_kernel=(21, 21)
            desaturate=True
            instances = outputs["instances"]
            if not instances.has("pred_masks"):
                return
            mask = instances.pred_masks.cpu().sum(0) >= 1
            mask = mask.numpy().astype("uint8")*255
            mask = np.stack([mask, mask, mask], axis=2)
            mask = cv2.GaussianBlur(mask, me_kernel, 0)
            foreground = image
            background = cv2.GaussianBlur(foreground, bg_kernel, 0)

            if desaturate:

                background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

            foreground = foreground.astype(float)
            background = background.astype(float)
            mask = mask.astype(float)/255.0
            foreground = cv2.multiply(foreground, mask)
            background = cv2.multiply(background, 1.0 - mask)
            vis_image = cv2.add(foreground, background)
            vis_image = Image.fromarray(np.uint8(vis_image[:, :, ::-1]))

    elif mode == 'grayscale':
        vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis_image = Image.fromarray(np.uint8(vis_image))

    elif mode == 'blur':
        vis_image = cv2.medianBlur(image, 5)
        vis_image = Image.fromarray(np.uint8(vis_image[:, :, ::-1]))

    elif mode == 'key_points':

        cfg = detectron.setup_cfg(config_file=model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"),
                                  weights_file=model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"),
                                  confidence_threshold=CONFIDENCE)
        predictor = DefaultPredictor(cfg)
        outputs = predictor(image)
        metadata_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        metadata_name = MetadataCatalog.get(metadata_name)
        visualizer = Visualizer(image[...,::-1], metadata_name)
        vis_image = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_image = Image.fromarray(np.uint8(vis_image.get_image()))

    elif mode == 'panoptic':

        cfg = detectron.setup_cfg(config_file=model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"),
                                  weights_file=model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"),
                                  confidence_threshold=CONFIDENCE)
        predictor = DefaultPredictor(cfg)
        pan_seg, seg_info = predictor(image)["panoptic_seg"]
        metadata_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        metadata_name = MetadataCatalog.get(metadata_name)
        visualizer = Visualizer(image[...,::-1], metadata_name)
        vis_image = visualizer.draw_panoptic_seg_predictions(pan_seg.to("cpu"), seg_info)
        vis_image = Image.fromarray(np.uint8(vis_image.get_image()))

    file_object = io.BytesIO()
    vis_image.save(file_object, 'JPEG')
    file_object.seek(0)
    os.remove(src)
    return send_file(file_object, mimetype='image/jpeg')


if __name__ == "__main__":

    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
