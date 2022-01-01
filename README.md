RUS | [ENG](#ENG)


<p align="center"><img src='https://i.ibb.co/rbJmBSV/Computer-Vision-Object-Detection-original.jpg'></p>
<a id='RUS'></a>
<a id='link6'></a>


Содержание:  

- [Введение](#link1)  
- [Локальная работа](#link2)  
   - [Скрипт process_img.py](#link3)  
   - [Скрипт process_video.py](#link4)   
- [Web-приложение](#link5) 
- [Ссылки](#linkru001)

*Примечание* Приложение в спящем режим. Для запуска потребуется время. Также, возможны сбои в работе - приложение на бесплатном сервере. 
Фото после загрузки удаляются. 


<h3>Введение</h3><a id='link1'></a>

Модульная обработка изображений с использованием [OpenCV](https://opencv.org/) и генераторов Python с помощью [Detectron2](https://github.com/facebookresearch/detectron2). Идея модульности позволяет программировать промышленный `pipeline` обработки несколькими специалистами DS. Идея использования генераторов взята у [Arosław Gilewski](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696) - в итоге получаем полноценный модуль, который может дополняться в нужных нам вариациях.  

Целью работы была реализация различных подходв внедрения Detectron2 и OpenCV в проекты - как локальные, так и web

*Примечание* Если у Вас не открывается ссылка на medium - включите режим инкогнито

1. **Локальное исполнение**. В данном исполнении - вы можете работать со случаными изображениями, либо применять обученную модель для детекции
оружия, в том числе на видео. При расширении обучающих данных, можно получить более точные результаты.
Интерфейс позволяет, не углубляясь в код, производить сегментацию изображений, в том числе и затенять фон.

2. **Web-приложение**. Реализовано дополнительно web-приложение с ограниченной функциональностью - Находится по адресу: https://detartyseg.herokuapp.com/. К сожалению - пришлось обрезать функционал, из-за ограничений размера на приложение. Также, воможен запуск через [Docker](https://www.docker.com/) - инструкция в соответствующей части файла [Readme.md](#link) - локально, либо после клонирования репозитория - запус через `app_local.py`  

----
[К содержанию](#link6)


<h3>Локальная работа</h3><a id='link2'></a>
<h4> 1. Скрипт process_img.py</h4><a id='link3'></a>

Здесь представлено три варианта работы модели:   

- `instance segmentation` на классы `COCO`
- `instance segmentation` для обнаружения оружия на кастомном [датасете](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar). Изображения были взяты с репозитория [MLWhiz](https://github.com/MLWhiz/object_detection), которые я разметил вручную с помощью [supervise.ly](https://supervise.ly/).    

Обучение вы можете посмотреть в этом [ноутбуке](https://nbviewer.org/github/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_detection_ipynb__.ipynb), если захотите обучить свой детектор (доступно и в [Collab](https://colab.research.google.com/drive/1TOSFHYrQrxToQ4v5azP8xzpuLcgEKrQ3?usp=sharing)). Веса базовых моделей можно брать из [зоопарка](https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py) - формировать `.yaml` аналогично представленным в папке `config`
- `separate background` - отделение фона, вместо аннотации. Идея предложена [Arosław Gilewski](https://medium.com/deepvisionguru/how-to-embed-detectron2-in-your-computer-vision-project-817f29149461)


Для удобства запуска, данному модулю было добавлено простое `GUI`, с помощью [GOOEY](https://github.com/chriskiehl/Gooey)

---
**Установка локально**  

Рекомендуется устанавливать в `virtuenv`. Необязательный шаг - можно пропустить

```
pip install virtualenv
virtualenv detection
source detection/bin/activate
```


Также рекомендуется дочитать инструкции до конца перед началом установки. Для `Win` рекомендуется использовать `Ubuntu 20.04.3 LTS` с `WLS2` или `Conda` терминал. Желательно, чтобы у вас уже стояла [Anaconda](https://www.anaconda.com/products/individual)


Шаги настройки модуля. 

```
git clone https://github.com/ArtyKrafty/detection
pip install -r requirements_loc.txt
```

Detectron2 необходимо установить в соседний каталог с папкой модуля

<img src="https://i.ibb.co/R25y4Lx/2021-12-26-19-35-04.png" alt="2021-12-26-19-35-04" border="0">

Для Win у вас должно быть установлено Visual C++ 2015 build tools   
Если нет - установите [отсюда](https://www.microsoft.com/ru-ru/download/confirmation.aspx?id=48159)  - это необходимо для [Detectron2](https://github.com/philferriere/cocoapi)


Также, рекомендуется проверить версию torch и torchvision - проверить можно [тут](https://pytorch.org/get-started/locally/) - 
выбираем Package - копируем команду для установки


<img src="https://i.ibb.co/wc1NPMq/123.jpg" alt="123" border="0">  


Если у вас появляется ошибка - `Torchvision error: Could not find module image.pyd` - рекомендую ознакомиться [здесь](https://kontext.tech/column/python/915/torchvision-error-could-not-find-module-imagepyd) с решением

Например, для `CPU`:

```
pip3 install torch torchvision torchaudio
или
conda install pytorch torchvision cpuonly -c pytorch

```

```
cd

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# На macOS, необходимо установить так (особенно, если у вас M1)
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install -e detectron2

Если возникнут сложности, можно обратиться к официальной документации

наиболее частая проблема - отсутствие С++ compiler 
conda install -c conda-forge compilers

```
[Официальное руководство по установке Detectron2](https://github.com/facebookresearch/detectron2/blob/3def12bdeaacd35c6f7b3b6c0097b7bc31f31ba4/INSTALL.md).   
[Официальное руководство по установке Detectron2. Installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

___

**Использование модуля**


1. `Instance segmentation` на классы `COCO`

```
Открываем терминал

cd detection
pythonw process_img.py

```

Обратите внимание, что вызов осуществляется через `pythonw`, не `python` или `python3`

У вас откроется окно:

<img src="https://i.ibb.co/ZJSL1yL/2021-12-28-10-43-20.png" alt="2021-12-28-10-43-20" border="0">


```
Необходимо задать input. Это изображения, на которых будут предсказываться объекты. 
Вы можете воспользоваться моими фотографиями - папка

./examples/images/photo/

также вы можете выбрать порог confidence. 
Нажимаем START и в папке outputs у вас обработанные изображения

```
<img src="https://i.ibb.co/pz0Twxx/0299ce77-30e4-4b31-b8e5-fad85b140111.jpg" alt="0299ce77-30e4-4b31-b8e5-fad85b140111" border="0">


2. `Instance segmentation` для обнаружения оружия на кастомном [датасете](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar). Названиями классов пришлось пренебречь - иначе потеряли в универсальности. Если собрать сет больше - можно использовать подобные модели для обнаружения
наличия оружия у человека

```
Открываем терминал

cd detection
pythonw process_img.py

```

Обратите внимание, что вызов осуществляется через `pythonw`, не `python` или `python3`

У вас откроется окно:

<img src="https://i.ibb.co/1TmBmfX/2021-12-28-10-10-42.png" alt="2021-12-28-10-10-42" border="0">


```
Необходимо задать input. Это изображения, на которых будут предсказываться объекты. 
Вы можете воспользоваться моими фотографиями - папка

./examples/images/weapon/

также вы можете выбрать порог confidence. Тут необходимо поменять модель - 
вы найдете нужную в папке config (и веса к ней):   


путь ./detection/configs/COCO-InstanceSegmentation_weapon/mask_rcnn_R_50_FPN_3x.yaml. 
Грузим к ней веса: ./detection/configs/COCO-InstanceSegmentation_weapon/model_final.pth
Нажимаем START и в папке outputs у вас обработанные изображения
```
<img src="https://i.ibb.co/8M9Bt3T/new-use-hotel-robber-with-gun-2.jpg" alt="new-use-hotel-robber-with-gun-2" border="0">

3. Отделение фона - работает аналогично - достаточно поставить отметку separate_background

Получаем изображение с blur-эффектом на фон

<img src="https://i.ibb.co/Q9m5nHh/004.jpg" alt="004" border="0">

----
[К содержанию](#link6)

<h4> 2. Скрипт process_video.py</h4><a id='link4'></a>

**Использование модуля**


`Instance segmentation` на классы `COCO`

```
Открываем терминал

cd detection
pythonw process_video.py

```

Обратите внимание, что вызов осуществляется через `pythonw`, не `python` или `python3`. Принцип работы аналогичный. Можно детектировать и оружие - 
по соображениям этики - было принято не добавлять файл с вооруженными нападениями в репозиторий

У вас откроется окно:

<img src="https://i.ibb.co/KKBhVRy/2021-12-27-23-26-40.png" alt="2021-12-27-23-26-40" border="0">

```
Необходимо задать input. Это видео, на котором будут предсказываться объекты. 
Вы можете воспользоваться моими примером

./examples/videos/cross_roads.mp4

также вы можете выбрать порог confidence. 
Нажимаем START и в папке outputs у вас обработанное видео. Занимает некоторое время

```
**Примечание.** Важно! На данный момент, существует [bug](https://github.com/facebookresearch/detectron2/issues/3780) на класс
`VideoVisualizer`. Чтобы его обойти, был использован класс `Visualizer`, который используется для изображений
поэтому маски от кадра к кадру меняются. Возможно bug наблюдается только на M1 silicone

![](sample.gif)
----
[К содержанию](#link6)

<h3>Web-приложение</h3><a id='link5'></a>
Скрипт app_local.py - для запуска локально

<a id='link'></a>

зеркало для app.py, который находится по адресу: https://detartyseg.herokuapp.com/. 
К сожалению - пришлось обрезать функционал, из-за ограничений размера на приложение

Если хотите запустить локально  - запускаем `app_local.py`:


```
 cd detection
 python app_local.py
 
```
Запустится приложение с ограниченным функционалом по адресу - `http://ВАШ_URL_из_терминала:8080/`


<img src="https://i.ibb.co/1LqDP6P/2021-12-30-22-41-29.png" alt="2021-12-30-22-41-29" border="0">

Здесь вы можете загрузить изображение с рабочего стола, выделить маски на нем или затемнить фон. Также возможно загрузка изображений по `URL` - только instance segmentation

Другой вариант - запуск через Docker (на М1 MAC OS данный способ может не сработать) - вес образа 14.4 GB:

```
cd detection

docker build . -f Dockerfile -t detectron2
docker run -d -p 8080:8080 detectron2


затем - после того, как закончили:

docker stop $(docker ps -a -q) 
docker rm $(docker ps -a -q) && docker rmi $(docker images | grep '^<none>' | awk '{print $3}')

```
----
[К содержанию](#link6)


<h3>Ссылки</h3><a id='linkru001'></a>

1. [Курс Deep learning school](https://www.dlschool.org/pro-track)  
2. [Arosław Gilewski](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696) - инженер CV  
3. [Detectron2](https://github.com/facebookresearch/detectron2) - библиотека Meta для работы с изображениями  
4. [OpenCV](https://opencv.org/) - средство обработки видео и изображений  
5. [Heroku](https://www.heroku.com/) - быстрый способ разработки и размещения приложений  
6. [pyTorch](https://pytorch.org/docs/stable/torch.html) - фреймворк работы с нейронными сетями  
7. [Docker](https://www.docker.com/) - средство упаковки и перемещения изображений  
8. [MLWhiz](https://github.com/MLWhiz/object_detection) - предоставлены образцы изображений для разметки

----
[К содержанию](#link6)

[RUS](#RUS) | ENG

<a id='ENG'></a>


<p align="center"><img src='https://i.ibb.co/rbJmBSV/Computer-Vision-Object-Detection-original.jpg'></p>
<a id='RUS'></a>
<a id='link7'></a>


Table of content:  

- [Introduction](#link8)  
- [Local installation](#link9)  
   - [Script process_img.py](#link10)  
   - [Script process_video.py](#link11)   
- [Web-application](#link12) 
- [Credits](#linkeng001)

*Note* Application is in sleep mode. It will take time to launch. Also, malfunctions are possible - the application is on a free server.
Photos are deleted after upload.

<h3>Introduction</h3><a id='link8'></a>

Modular image processing using [OpenCV](https://opencv.org/) and Python generators using [Detectron2](https://github.com/facebookresearch/detectron2). The idea of modularity allows the industrial processing pipeline to be programmed by several DS specialists. The idea of using generators is taken from [Arosław Gilewski](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696) - as a result, we get a full-fledged module that can be supplemented in the variations we need.

The aim of the work was to implement various approaches to integrating Detectron2 and OpenCV into projects - both local and web

*Note* If you do not have a link to medium - turn on incognito mode

1. **Local execution**. In this version, you can work with random images, or use a trained model for detection
weapons, including video. By expanding the training data, you can get more accurate results.
The interface allows, without delving into the code, to perform image segmentation, including shading the background.

2. **Web Application**. An additional web application with limited functionality has been implemented - Located at: https://detartyseg.herokuapp.com/. Unfortunately, I had to cut the functionality due to the size restrictions for the application. Also, launching via [Docker](https://www.docker.com/) is possible - instructions in the corresponding part of the file [Readme.md](#linkdock) - locally, or after cloning the repository - launching via `app_local.py`

----
[Back to contest](#link7)


<h3>Local execution</h3><a id='link9'></a>
<h4> 1. Script process_img.py</h4><a id='link10'></a>

Here are three options for how the model works:

- `instance segmentation` into classes` COCO`
- `instance segmentation` for detecting weapons on a custom [dataset](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar). The images were taken from the [MLWhiz](https://github.com/MLWhiz/object_detection) repository, which I mapped out manually using [supervise.ly](https://supervise.ly/).

You can watch the training in this [notebook](https://nbviewer.org/github/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_detection_ipynb__.ipynb), if you want to train your detector (available in [Collab](https://colab.research.google.com/drive/1TOSFHYrQrxToQ4v5azP8xzpuLcgEKrQ3?usp=sharing)). The weights of the base models can be taken from the [zoo](https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py) - form `.yaml` similarly to those presented in the` config` folder
- `separate background` - separation of the background, instead of annotation. Idea suggested by [Arosław Gilewski](https://medium.com/deepvisionguru/how-to-embed-detectron2-in-your-computer-vision-project-817f29149461)


For ease of launch, a simple `GUI` was added to this module, using [GOOEY](https://github.com/chriskiehl/Gooey)

---
**Install locally**

It is recommended to install to `virtuenv`. Optional step - can be skipped

```
pip install virtualenv
virtualenv detection
source detection/bin/activate
```


It is also recommended to read the instructions to the end before starting the installation. For `Win` it is recommended to use` Ubuntu 20.04.3 LTS` with `WLS2` or` Conda` terminal. It is desirable that you already have [Anaconda](https://www.anaconda.com/products/individual)


Module configuration steps.

```
git clone https://github.com/ArtyKrafty/detection
pip install -r requirements_loc.txt
```

Detectron2 must be installed in the adjacent directory with the module folder

<img src="https://i.ibb.co/R25y4Lx/2021-12-26-19-35-04.png" alt="2021-12-26-19-35-04" border="0">

For Win, you must have Visual C ++ 2015 build tools installed
If not, install [from here](https://www.microsoft.com/en-ru/download/confirmation.aspx?id=48159) - this is required for [Detectron2](https://github.com/philferriere/cocoapi)


Also, it is recommended to check the version of torch and torchvision - you can check [here](https://pytorch.org/get-started/locally/) - select Package - copy the command to install


<img src="https://i.ibb.co/wc1NPMq/123.jpg" alt="123" border="0">  


If you get an error - `Torchvision error: Could not find module image.pyd` - I recommend reading [here](https://kontext.tech/column/python/915/torchvision-error-could-not-find-module-imagepyd) with solution

For example, for `CPU`:

```
pip3 install torch torchvision torchaudio
или
conda install pytorch torchvision cpuonly -c pytorch

```

```
cd

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you need to install like this (especially if you have M1
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install -e detectron2

If you have any difficulties, you can refer to the official documentation

the most common problem is the lack of a C ++ compiler:
conda install -c conda-forge compilers

```
[Detectron2 Official Installation Guide](https://github.com/facebookresearch/detectron2/blob/3def12bdeaacd35c6f7b3b6c0097b7bc31f31ba4/INSTALL.md).  
[Official Installation Guide for Detectron2. Installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

___

**Module usage**


1. `Instance segmentation` with `COCO`

```
Open terminal

cd detection
pythonw process_img.py

```

Note that the call is made through `pythonw`, not` python` or `python3`

A window will open:

<img src="https://i.ibb.co/ZJSL1yL/2021-12-28-10-43-20.png" alt="2021-12-28-10-43-20" border="0">


```
You must set input. These are the images on which objects will be predicted.
You can use my photos - folder

./examples/images/photo/

also you can choose the confidence threshold.
Press START and in the outputs folder you have processed images
```
<img src="https://i.ibb.co/pz0Twxx/0299ce77-30e4-4b31-b8e5-fad85b140111.jpg" alt="0299ce77-30e4-4b31-b8e5-fad85b140111" border="0">


2. `Instance segmentation` for detecting weapons on a custom [dataset](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar). The class names had to be neglected - otherwise they lost in universality. If you collect a larger set, you can use similar models to detect whether a person has a weapon

```
Open terminal

cd detection
pythonw process_img.py

```

Note that the call is made through `pythonw`, not` python` or `python3`

A window will open:

<img src="https://i.ibb.co/1TmBmfX/2021-12-28-10-10-42.png" alt="2021-12-28-10-10-42" border="0">


```
You must set input. These are the images on which objects will be predicted.
You can use my photos - folder

./examples/images/weapon/

also you can choose the confidence threshold. Here you need to change the model -
you will find the needed one in the config folder (and weights to it):


path ./detection/configs/COCO-InstanceSegmentation_weapon/mask_rcnn_R_50_FPN_3x.yaml.
We load weights to it: ./detection/configs/COCO-InstanceSegmentation_weapon/model_final.pth
Press START and in the outputs folder you have processed images
```
<img src="https://i.ibb.co/8M9Bt3T/new-use-hotel-robber-with-gun-2.jpg" alt="new-use-hotel-robber-with-gun-2" border="0">

3. Separating the background - works the same way - just check the separate_background

Getting an image with a blur effect on the background

<img src="https://i.ibb.co/Q9m5nHh/004.jpg" alt="004" border="0">

----
[Back to contest](#link7)

<h4> 2. Script process_video.py</h4><a id='link11'></a>

**Module usage**


`Instance segmentation` with `COCO`

```
Open terminal

cd detection
pythonw process_video.py

```

Note that the call is made through `pythonw`, not` python` or `python3`. The principle of operation is similar. Weapons can also be detected -
for ethical reasons - it was customary not to add the file with armed attacks to the repository

A window will open:

<img src="https://i.ibb.co/KKBhVRy/2021-12-27-23-26-40.png" alt="2021-12-27-23-26-40" border="0">

```
You must set input. This is a video in which objects will be predicted.
You can use my example

./examples/videos/cross_roads.mp4

also you can choose the confidence threshold.
Press START and in the outputs folder you have the processed video. Takes some time

```
**Note.** Important! At the moment, there is a [bug](https://github.com/facebookresearch/detectron2/issues/3780) for the class
`VideoVisualizer`. To get around it, the `Visualizer` class was used, which is used for images
therefore the masks change from frame to frame. Probably the bug is only observed on M1 silicone

![](sample.gif)
----
[Back to contest](#link7)

<h3>Web-application</h3><a id='link12'></a>
Script app_local.py - for local work

<a id='linkdock'></a>

mirror for app.py, which is located at: https://detartyseg.herokuapp.com/.
Unfortunately - I had to cut the functionality, due to the size restrictions for the application

If you want to run locally, run `app_local.py`:


```
 cd detection
 python app_local.py
 
```

An application with limited functionality will start at the address - `http: // YOUR_URL_from_terminal: 8080 /`

<img src="https://i.ibb.co/1LqDP6P/2021-12-30-22-41-29.png" alt="2021-12-30-22-41-29" border="0">

Here you can load an image from the desktop, select masks on it or darken the background. It is also possible to download images by `URL` - only instance segmentation

Another option is to run via Docker (on M1 MAC OS, this method may not work) - weight of Image 14.4 GB:

```
cd detection

docker build . -f Dockerfile -t detectron2
docker run -d -p 8080:8080 detectron2



then - after finished:

docker stop $(docker ps -a -q) 
docker rm $(docker ps -a -q) && docker rmi $(docker images | grep '^<none>' | awk '{print $3}')

```
----
[Back to contest](#link7)


<h3>Credits</h3><a id='linkeng001'></a>

1. [Deep learning school](https://www.dlschool.org/pro-track)  
2. [Arosław Gilewski](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696) - CV engineer  
3. [Detectron2](https://github.com/facebookresearch/detectron2) - Meta library for working with images  
4. [OpenCV](https://opencv.org/) - video and image processing tool  
5. [Heroku](https://www.heroku.com/) - a fast way to develop and host applications  
6. [pyTorch](https://pytorch.org/docs/stable/torch.html) - framework for working with neural networks  
7. [Docker](https://www.docker.com/) - packaging and image transfer tool
8. [MLWhiz](https://github.com/MLWhiz/object_detection) - provided sample images for markup


----
[Back to contest](#link7)
