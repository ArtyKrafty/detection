import os
from tqdm import tqdm
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

@Gooey(progress_regex=r"^progress: (?P<current>\d+)/(?P<total>\d+)$",
       progress_expr="current / total * 100",
       timing_options = {
        'show_time_remaining':True,
        'hide_time_remaining_on_complete':True,
    })
def parse_args():
    import argparse

    # Аргументы командной строки
    ap = GooeyParser(description="Detectron2 пайплайн обработки изображений")
    ap.add_argument("-i", "--input", default="examples/images/photo/", required=True,
                    help="путь к папке с изображениями. Примеры в папке examples", widget='DirChooser')
    ap.add_argument("-o", "--output", default="output",
                    help="путь для сохранения результатов (по-умолчанию: output)", widget='DirChooser')
    ap.add_argument("-p", "--progress", action="store_true",
                    help="показывать прогресс обработки")
    ap.add_argument("-sb", "--separate-background", action="store_true",
                    help="отделить фон")

    # Detectron2 гиперпараметры
    ap.add_argument("--config-file",
                    default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                    help="путь к файлу с настройками модели в папке configs (по-умолчанию, COCO-InstanceSegmentation." 
                    "Веса задавать не надо. Для оружия - COCO-InstanceSegmentation_weapon)", widget='FileChooser')
    ap.add_argument("--weights", default=None,
                    help="путь к файлу с весами .pth", widget='FileChooser')
    ap.add_argument("--confidence-threshold", type=float, default=0.7,
                    help="confidence-порог (по-умолчанию: 0.7)")

    return ap.parse_args()


def main(args):
    # Создаем директорию для результатов модели
    os.makedirs(args.output, exist_ok=True)

    # Создаем шаги пайплайна
    capture_images = CaptureImgs(args.input) \
        if os.path.isdir(args.input) else CaptureImg(args.input)

    cfg = detectron.setup_cfg(config_file=args.config_file,
                              weights_file=args.weights,
                              confidence_threshold=args.confidence_threshold)

    predict = Predict(cfg)

    if args.separate_background:
        separate_background = Separate("vis_image")
        annotate_image = None
    else:
        separate_background = None
        metadata_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        metadata_name = MetadataCatalog.get(metadata_name)
        annotate_image = AnnotateImg("vis_image", metadata_name)
        
    save_image = SaveImg("vis_image", args.output)

    pipeline = (capture_images |
                predict |
                separate_background |
                annotate_image |
                save_image)

    try:
        for _ in tqdm(pipeline, disable=not args.progress):
            pass
    except StopIteration:
        return
    except KeyboardInterrupt:
        return



if __name__ == "__main__":
    
    args = parse_args()
    main(args)