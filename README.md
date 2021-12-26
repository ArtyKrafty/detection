Modular image processing pipeline using OpenCV and Python generators powered by Detectron2.



$ conda env create -f environment.yml
$ conda activate detectron2-pipeline



$ cd ..
$ git clone https://github.com/facebookresearch/detectron2.git
$ cd detectron2
$ git checkout 3def12bdeaacd35c6f7b3b6c0097b7bc31f31ba4
$ python setup.py build develop



$ python process_images.py -i assets/images/friends -p

$ python process_video.py -i assets/videos/walk.small.mp4 -p -d -ov walk.small.mp4