import os
from tqdm import tqdm
from gooey import Gooey, GooeyParser
from detection.capture_video import CaptureVideo
from detection.predict import Predict
from detection.annotate_video import AnnotateVideo
from detection.display_video import DisplayVideo
from detection.save_video import SaveVideo
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

    ap = GooeyParser(description="Detectron2 пайплайн обработки изображений")
    ap.add_argument("-i", "--input", default="examples/videos/cross_roads.mp4",
                    help="путь к видеофайлу", widget='FileChooser', required=True)
    ap.add_argument("-o", "--output", default="output",
                    help="путь для сохранения результатов (по-умолчанию: output)", widget='DirChooser')
    ap.add_argument("-ov", "--out-video", default="sample.mp4",
                    help="наименование итогового файла")
    ap.add_argument("-p", "--progress", action="store_true",
                    help="показывать прогресс обработки")
    ap.add_argument("-d", "--display", action="store_true",
                    help="показывать видео")



    # Detectron2 гиперпараметры
    ap.add_argument("--config-file",
                    default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                    help="путь к файлу с настройками модели в папке configs. На других сейчас наблюдается ошибка при отрисовке: "\
                    "на M1 - bug # 3780." 
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
    if args.input.isdigit():
        capture_video = CaptureVideo(int(args.input))
    elif os.path.isfile(args.input):
        capture_video = CaptureVideo(args.input)


    cfg = detectron.setup_cfg(config_file=args.config_file,
                              weights_file=args.weights,
                              confidence_threshold=args.confidence_threshold)


    predict = Predict(cfg)


    metadata_name = cfg.DATASETS.TEST[0]
    metadata = MetadataCatalog.get(metadata_name)
    
    annotate_video = AnnotateVideo("vis_image", metadata)

    display_video = DisplayVideo("vis_image") \
        if args.display else None

    save_video = SaveVideo("vis_image", os.path.join(args.output, args.out_video)) if args.out_video else None

    pipeline = (capture_video |
                predict |
                annotate_video |
                display_video |
                save_video)


    try:
        for _ in tqdm(pipeline,
                      total=capture_video.frame_count if capture_video.frame_count > 0 else None,
                      disable=not args.progress):
            pass
    except StopIteration:
        return
    except KeyboardInterrupt:
        return
    finally:
        if isinstance(predict, CaptureVideo):
            capture_video.cleanup()
        if display_video:
            display_video.cleanup()
        if save_video:
            save_video.cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)