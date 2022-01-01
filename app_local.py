import requests
import os
import cv2
import numpy as np
import io
from tqdm import tqdm
from PIL import Image
from detection.utils import detectron
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from flask import Flask, render_template, request, send_file
from detectron2.engine.defaults import DefaultPredictor

# скрипт app_local.py нужен для работы web-приложения на локальной машине

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/client/img/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resizing(img, new_width=None, new_height=None, interp=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    if new_width is None and new_height is None:
        return img
    if new_width is None:
        ratio = new_height / height
        dimension = (int(width * ratio), new_height)
    else:
        ratio = new_width / width
        dimension = (new_width, int(height * ratio))
    res_img = cv2.resize(img, dimension, interpolation=interp)
    return res_img

@app.route("/")
def home():
    return render_template('home.html')


@app.route("/predict", methods=['POST', 'GET'])
def main():
    try: 
        

        method = request.method
        if method == 'GET':
            url = request.args.get("url")
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content))
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], "image_001.jpeg"))
            src = os.path.join(app.config['UPLOAD_FOLDER'], "image_001.jpeg")
            mode = "instance_segmentation"
            CONFIDENCE = .7
        elif method == 'POST':
            try:
                file = request.files['file']

                if file and allowed_file(file.filename):
                    filename = file.filename
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    src = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    mode = request.form["mode"]
                    CONFIDENCE = float(request.form["confidence"])

            except:
                return render_template("error.html")

        image = cv2.imread(src)
        if mode in ['separate_background', 'instance_segmentation']:
            cfg = detectron.setup_cfg(config_file="./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                                      weights_file=None,
                                      confidence_threshold=CONFIDENCE)
            predictor = DefaultPredictor(cfg)
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

        file_object = io.BytesIO()
        vis_image.save(file_object, 'JPEG')
        file_object.seek(0)
        os.remove(src)
        return send_file(file_object, mimetype='image/jpeg')
    except:
        return render_template("error.html")

if __name__ == "__main__":

    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)