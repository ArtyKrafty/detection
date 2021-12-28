
import requests
import os
import io
from tqdm import tqdm
from PIL import Image
from gooey import Gooey, GooeyParser
from detection.capture_imgs import CaptureImgs
from detection.capture_img import CaptureImg
from detection.predict import Predict
from detection.save_img import SaveImg
from detection.annotate_img import AnnotateImg
from detection.separate_background import Separate
from detection.utils import detectron
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from flask import Flask, render_template, request, send_from_directory, send_file



web_process_img = Flask(__name__)
detector = Detector()
def load_image(url):
	response = requests.get(url)
	img = Image.open(io.BytesIO(response.content))
	return img


def run_inference(img_path = 'file.jpg'):
	result_img = detector.inference(img_path)
	try:
		os.remove(img_path)
	except:
		pass

	return result_img


@web_process_img.route("/")
def index():
	return render_template('index.html')


@web_process_img.route("/detect", methods=['POST', 'GET'])
def upload():
	if request.method == 'POST':
		try:
			file = Image.open(request.files['file'].stream)
			rgb_im = file.convert('RGB')
			rgb_im.save('file.jpg')
		except:
			return render_template("failure.html")
	elif request.method == 'GET':
		url = request.args.get("url")
		try:
			rgb_im = load_image_url(url)
			rgb_im = rgb_im.convert('RGB')
			rgb_im.save('file.jpg')
		except:
			return render_template("failure.html")
	result_img = run_inference('file.jpg')
	file_object = io.BytesIO()
	result_img.save(file_object, 'PNG')
	file_object.seek(0)

	return send_file(file_object, mimetype='image/jpeg')

if __name__ == "__main__":

	port = int(os.environ.get('PORT', 8080))
	app.run(host='0.0.0.0', port=port)