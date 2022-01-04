RUS | [ENG](#ENG)

<a id='RUS'></a>
<h2 align="center">Модульная обработка изображений с использованием OpenCV и генераторов Python с помощью Detectron2. Pipeline, web-приложение и локальное приложение </h2>

<p align="center"><img src='https://i.ibb.co/rbJmBSV/Computer-Vision-Object-Detection-original.jpg'></p>

<a id='link6'></a>



Содержание:  

 [Введение](#link1)  
- [Локальная работа](#link2)  
   - [Скрипт process_img.py](#link3)  
   - [Скрипт process_video.py](#link4)   
- [Web-приложение](#link5) 
- [Dockerfile](#link)
- [Техническая поддержка](#linkprobru)
- [Ссылки](#linkru001)


---

📝 Приложение в спящем режим. Для запуска потребуется время. Также, возможны сбои в работе - приложение на бесплатном сервере. 
Фото после загрузки удаляются. 


<p align="center"><img src="https://i.ibb.co/X4W8wCw/2022-01-03-02-23-14.png" alt="2022-01-03-02-23-14" border="0"></p>


📝 Ошибка может быть связана с превышением памяти. Можно попробовать еще раз или взять другую картинку. С локальной сборкой проблем не возникнет

---


<h3>Введение</h3><a id='link1'></a>

Модульная обработка изображений с использованием [OpenCV](https://opencv.org/) и генераторов `Python` с помощью [Detectron2](https://github.com/facebookresearch/detectron2). Идея модульности позволяет программировать промышленный `pipeline` обработки несколькими специалистами `DS`. Идея использования генераторов взята у [Arosław Gilewski](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696) - в итоге получаем полноценный модуль, который может дополняться в нужных нам вариациях.  

Целью работы была реализация различных подходв внедрения `Detectron2` и `OpenCV` в проекты - как локальные, так и `web` с использованием модулей 
или же в виде скриптов с применением различных вариантов работы моделей

> 📝 Если у Вас не открывается ссылка на `medium` - включите режим инкогнито

1. **Локальное исполнение**. В данном исполнении - вы можете работать со случаными изображениями, либо применять обученную модель для детекции
оружия, в том числе на видео. При расширении обучающих данных, можно получить более точные результаты.
Интерфейс позволяет, не углубляясь в код, производить сегментацию изображений, в том числе и затенять фон.

   Доступный функционал:  

      - Изображения:
        - Instance segmentation на изображениях
        - Keypoints
        - Затемнение фона (separate background)
        - Panoptic segmentation
        - Детекция оружия (custom dataset)
      - Видео:
        - Instance segmentation на видео
        - Показ обработки в режиме реального времени
        - Детекция оружия на видео
        
        
        
  <p align="center"><img src="https://i.ibb.co/ZJSL1yL/2021-12-28-10-43-20.png" alt="2021-12-28-10-43-20" border="0"></p>
  
  
 > 📝 Подойдет тем, кто хочет иметь доступ к коду и иметь больший контроль. 
  
2. **Web-приложение**. Реализовано дополнительно `web-приложение` с ограниченной функциональностью (только изображения) - Находится по адресу: https://detartyseg.herokuapp.com/. К сожалению - пришлось обрезать функционал, из-за ограничений размера на приложение. Также, возможен запуск через [Docker](https://www.docker.com/) - инструкция в соответствующей части файла [Readme.md](#link) - локально, либо после клонирования репозитория и настройки окружения - запус через `app_local.py`  

   Доступный функционал:  

      - Изображения:
        - Instance segmentation на изображениях
        - Keypoints на изображениях
        - Panoptic segmentation на изображениях (`УДАЛЕНО из web. Доступно в docker`)
        - Затемнение фона (separate background)
        - Blur эффект
        - Черно-белое изображение


<p align="center"><img src="https://i.ibb.co/mDCxTP3/2022-01-03-11-27-44.png" alt="2022-01-03-11-27-44" border="0"></p>
 
> 📝 Подойдет тем, кто не хочет тратить время на настройки - а сразу предпочитает получить результат. Есть альтернатива в виде [Docker](#link) сборки

----
[К содержанию](#link6)


<h3>Локальная работа</h3><a id='link2'></a>
<h4> 1. Скрипт process_img.py</h4><a id='link3'></a>

Здесь представлено три варианта работы модели:   

- `instance segmentation` на классы `COCO`
- `instance segmentation` для обнаружения оружия на кастомном [датасете](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar). Изображения были взяты с репозитория [MLWhiz](https://github.com/MLWhiz/object_detection), которые я разметил вручную с помощью [supervise.ly](https://supervise.ly/).    

Обучение вы можете посмотреть в этом [ноутбуке](https://nbviewer.org/github/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_detection_ipynb__.ipynb), если захотите обучить свой детектор (доступно и в [Collab](https://colab.research.google.com/drive/1TOSFHYrQrxToQ4v5azP8xzpuLcgEKrQ3?usp=sharing)). Веса базовых моделей можно брать из [зоопарка](https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py) - формировать `.yaml` аналогично представленным в папке `config`
- `separate background` - отделение фона, вместо аннотации. Идея предложена [Arosław Gilewski](https://medium.com/deepvisionguru/how-to-embed-detectron2-in-your-computer-vision-project-817f29149461)
- `keypoints` - для построения ключевых точек людей на изображениях (`pose-flow`)
- `panoptic segmentation` - для паноптической сегментации изображений


Для удобства запуска, данному модулю было добавлено простое `GUI`, с помощью [GOOEY](https://github.com/chriskiehl/Gooey)

---

**Установка локально**  

Рекомендуется устанавливать в `virtuenv`. Необязательный шаг - можно пропустить

```python
pip install virtualenv
virtualenv detection
source detection/bin/activate
```


Также рекомендуется дочитать инструкции до конца перед началом установки. Для `Win` рекомендуется использовать `Ubuntu 20.04.3 LTS` с `WLS2` или `Conda` терминал. Желательно, чтобы у вас уже стояла [Anaconda](https://www.anaconda.com/products/individual)


Шаги настройки модуля. 

```python
git clone https://github.com/ArtyKrafty/detection
pip install -r requirements_loc.txt
```

Detectron2 необходимо установить в соседний каталог с папкой модуля

<p align="center"><img src="https://i.ibb.co/R25y4Lx/2021-12-26-19-35-04.png" alt="2021-12-26-19-35-04" border="0"></p>

Для Win у вас должно быть установлено Visual C++ 2015 build tools   
Если нет - установите [отсюда](https://www.microsoft.com/ru-ru/download/confirmation.aspx?id=48159)  - это необходимо для [Detectron2](https://github.com/philferriere/cocoapi)


Также, рекомендуется проверить версию torch и torchvision - проверить можно [тут](https://pytorch.org/get-started/locally/) - 
выбираем Package - копируем команду для установки


<p align="center"><img src="https://i.ibb.co/wc1NPMq/123.jpg" alt="123" border="0"></p> 


> 📝 Если у вас появляется ошибка - `Torchvision error: Could not find module image.pyd` - рекомендую ознакомиться [здесь](https://kontext.tech/column/python/915/torchvision-error-could-not-find-module-imagepyd) с решением

Например, для `CPU`:

```python
pip3 install torch torchvision torchaudio
или
conda install pytorch torchvision cpuonly -c pytorch

```
Установка `Detectron2`

```python
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

```python
Открываем терминал

cd detection
pythonw process_img.py

```

> 📝 Обратите внимание, что вызов осуществляется через `pythonw`, не `python` или `python3`

У вас откроется окно:

<p align="center"><img src="https://i.ibb.co/ZJSL1yL/2021-12-28-10-43-20.png" alt="2021-12-28-10-43-20" border="0"></p>


```python
Необходимо задать input. Это изображения, на которых будут предсказываться объекты. 
Вы можете воспользоваться моими фотографиями - папка

./examples/images/photo/

также вы можете выбрать порог confidence. 
Нажимаем START и в папке outputs у вас обработанные изображения

```
<p align="center"><img src="https://i.ibb.co/pz0Twxx/0299ce77-30e4-4b31-b8e5-fad85b140111.jpg" alt="0299ce77-30e4-4b31-b8e5-fad85b140111" border="0"></p>


2. `Instance segmentation` для обнаружения оружия на кастомном [датасете](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar). Названиями классов пришлось пренебречь - иначе потеряли в универсальности. Если собрать сет больше - можно использовать подобные модели для обнаружения
наличия оружия у человека

```python
Открываем терминал

cd detection
pythonw process_img.py

```

> 📝 Обратите внимание, что вызов осуществляется через `pythonw`, не `python` или `python3`

У вас откроется окно:

<p align="center"><img src="https://i.ibb.co/1TmBmfX/2021-12-28-10-10-42.png" alt="2021-12-28-10-10-42" border="0"></p>


```python
Необходимо задать input. Это изображения, на которых будут предсказываться объекты. 
Вы можете воспользоваться моими фотографиями - папка

./examples/images/weapon/

также вы можете выбрать порог confidence. Тут необходимо поменять модель - 
вы найдете нужную в папке config (и веса к ней):   


путь ./detection/configs/COCO-InstanceSegmentation_weapon/mask_rcnn_R_50_FPN_3x.yaml. 
Грузим к ней веса: ./detection/configs/COCO-InstanceSegmentation_weapon/model_final.pth
Нажимаем START и в папке outputs у вас обработанные изображения
```
<p align="center"><img src="https://i.ibb.co/8M9Bt3T/new-use-hotel-robber-with-gun-2.jpg" alt="new-use-hotel-robber-with-gun-2" border="0"></p>

3. Отделение фона - работает аналогично - достаточно поставить отметку separate_background

```python
Необходимо задать input. Это изображения, на которых будут предсказываться объекты. 
Вы можете воспользоваться моими фотографиями - папка

./examples/images/photos/

также вы можете выбрать порог confidence. Тут необходимо поменять модель - 
вы найдете нужную в папке config (и веса к ней):   


путь ./detection/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml. 
Ставим отметку - separate background:
Нажимаем START и в папке outputs у вас обработанные изображения

```

Получаем изображение с blur-эффектом на фон - инвертируем маску и вместо предсказаний получаем выделенный от фона объектр

<p align="center"><img src="https://i.ibb.co/Q9m5nHh/004.jpg" alt="004" border="0"></p>

4. Построение ключевых точек:

```python
Необходимо задать input. Это изображения, на которых будут предсказываться объекты. 
Вы можете воспользоваться моими фотографиями - папка

./examples/images/key_points/

также вы можете выбрать порог confidence. Тут необходимо поменять модель - 
вы найдете нужную в папке config (и веса к ней):   


путь ./detection/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml. 
Нажимаем START и в папке outputs у вас обработанные изображения

Веса задавать не надо. 

```
Получаем изображение с отмеченными ключевыми точками. Предназначено для работы с изображениями с людьми. 


<p align="center"><img src="https://i.ibb.co/0KY5Jh1/predict.jpg" alt="predict" border="0"></p>

5. Паноптическая сегментация изображений - достаточно указать адрес к конфигурации
```python
Необходимо задать input. Это изображения, на которых будут предсказываться объекты. 
Вы можете воспользоваться моими фотографиями - папка

./examples/images/pan/

также вы можете выбрать порог confidence. Тут необходимо поменять модель - 
вы найдете нужную в папке config (и веса к ней):   


путь ./detection/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml 
Нажимаем START и в папке outputs у вас обработанные изображения

Веса задавать не надо. 

```

<p align="center"><img src="https://i.ibb.co/Zx9zB1g/predict.jpg" alt="predict" border="0"></p>


Целью было продемонстрировать различные варианты работы модели без углублений в код со стороны пользователя данного модуля

----
[К содержанию](#link6)

<h4> 2. Скрипт process_video.py</h4><a id='link4'></a>

**Использование модуля**


`Instance segmentation` на классы `COCO`

```python
Открываем терминал

cd detection
pythonw process_video.py

```

> 📝 Обратите внимание, что вызов осуществляется через `pythonw`, не `python` или `python3`. Принцип работы аналогичный. Можно детектировать и оружие - 
> по соображениям этики - было принято не добавлять файл с вооруженными нападениями в репозиторий

У вас откроется окно:

<p align="center"><img src="https://i.ibb.co/KKBhVRy/2021-12-27-23-26-40.png" alt="2021-12-27-23-26-40" border="0"></p>

```python
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


Скрипт `app_local.py` - для запуска локально



зеркало для `app.py`, который находится по адресу: https://detartyseg.herokuapp.com/. 
К сожалению - пришлось обрезать функционал, из-за ограничений размера на приложение. Пришлось убрать паноптическую сегментацию -
превышения памяти на приложение, на бесплатном сервере. Остался функционал в локальном приложении и `docker`

Если хотите запустить локально  - запускаем `app_local.py`:


```python
 git clone https://github.com/ArtyKrafty/detection
 cd detection
 python app_local.py
 
```
Запустится приложение с ограниченным функционалом по адресу - `http://ВАШ_URL_из_терминала:8080/`


<p align="center"><img src="https://i.ibb.co/mDCxTP3/2022-01-03-11-27-44.png" alt="2022-01-03-11-27-44" border="0"></p>

Здесь вы можете загрузить изображение с рабочего стола, выделить маски на нем или затемнить фон. Также возможно загрузка изображений по `URL` - только instance segmentation

<a id='link'></a>

Другой вариант - запуск через `Docker` (локальная версия https://detartyseg.herokuapp.com/) - вес образа `~14.1 GB` (у вас должен быть установлен
[Docker](https://www.docker.com/) - дополнительно ничего не требуется):

```python
git clone https://github.com/ArtyKrafty/detection

cd detection

docker build . -f Dockerfile -t detectron2
docker run --name detectron2 -p 127.0.0.1:8080:8080 detectron2


затем - после того, как закончили:

docker stop detectron2
docker rm $(docker ps -qa)

или использовать docker-desktop приложение, что может быть удобнее

посмотреть директории внутри: docker run -t -i detectron2 /bin/bash
```
<p align="center"><img src="https://i.ibb.co/FwgVWJn/123.jpg" alt="123" border="0"></p>

> 📝 Если у вас `MacOS` - зайдите в настройки `docker -> advanced` и увеличьте память `RAM` с 2 до 4 ГБ - иначе `С++` компилятор 
не сможет собрать `torch`

----
[К содержанию](#link6)

<h3>Техническая поддержка</h3><a id='linkprobru'></a>

Проблемы возникают не с приложением, а могут возникнуть при установки сопутствующего окружения - `Detectron2`, `torch` и так далее. 
Здесь собрал решения, которые помогут вам в установке, если вы решите работать не через `docker`. Тестировалось на `MacOs` и `Windows` - на `Win` - основная проблема - отсутствие `Microsoft Visual C++`. Данный сборник поможет решить основные проблемы по установке `torch` и `Detectron2`. Также, рекомендую, чтобы у вас уже была `Anaconda`

| Источник  |  Описание |  Решение |
|---|---|---|
| Приложение Web-app  | Появляется ошибка `heroku` - `Application Error`  | Ошибка может быть связана с превышением памяти. Можно попробовать еще раз выполнить действие или взять другую картинку. С локальной сборкой проблем не возникнет  |
|  torch, Detectron | C++ ошибка на Windows при установке на `pycatools`  |  У вас не установлен Visual Studio C++ - отсутствует компилятор. Установите [отсюда](https://www.microsoft.com/ru-ru/download/confirmation.aspx?id=48159) |
| torch, torchvision  | устаревшая версия или установлена не соответствющая вашему ПК  | проверить можно [тут](https://pytorch.org/get-started/locally/)|
| torch  | `Torchvision error: Could not find module image.pyd`  | ознакомиться [здесь](https://kontext.tech/column/python/915/torchvision-error-could-not-find-module-imagepyd) с решением  |
| Docker  | ошибка с `С++` при сборке или `os` | зайдите в настройки `docker -> advanced` и увеличьте память `RAM` с 2 до 4 ГБ|

----
[К содержанию](#link6)

<h3>Ссылки</h3><a id='linkru001'></a>

1. [Курс Deep learning school](https://www.dlschool.org/pro-track) - обучают и направляют. 
2. [Arosław Gilewski](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696) - инженер CV. Идея использования генераторов python и за baseline удаления фона. 
3. [Detectron2](https://github.com/facebookresearch/detectron2) - библиотека Meta для работы с изображениями с огромным зоопарком обученных моделей.   
4. [OpenCV](https://opencv.org/) - средство обработки видео и изображений. Без этой библиотеки было бы сложнее. 
5. [Heroku](https://www.heroku.com/) - быстрый способ разработки и размещения приложений. 
6. [pyTorch](https://pytorch.org/docs/stable/torch.html) - фреймворк работы с нейронными сетями  
7. [Docker](https://www.docker.com/) - средство упаковки и перемещения изображений  
8. [MLWhiz](https://github.com/MLWhiz/object_detection) - предоставлены образцы изображений для разметки

----
[К содержанию](#link6)

[RUS](#RUS) | ENG

<a id='ENG'></a>

<h2 align="center">Modular image and video processing with OpenCV and Detectron2. Pipeline, web-app and local-app </h2>


<p align="center"><img src='https://i.ibb.co/rbJmBSV/Computer-Vision-Object-Detection-original.jpg'></p>
<a id='RUS'></a>
<a id='link7'></a>


Table of content:  

- [Introduction](#link8)  
- [Local installation](#link9)  
   - [Script process_img.py](#link10)  
   - [Script process_video.py](#link11)   
- [Web-application](#link12) 
- [Dockerfile](#linkdock)
- [Help desk](#linkprobeng)
- [Credits](#linkeng001)


---


📝 Application is in sleep mode. It will take time to launch. Also, malfunctions are possible - the application is on a free server.
Photos are deleted after upload.

<p align="center"><img src="https://i.ibb.co/X4W8wCw/2022-01-03-02-23-14.png" alt="2022-01-03-02-23-14" border="0"></p>


📝 The error may be related to out of memory. You can try again or take another picture. There will be no problems with local assembly

---


<h3>Introduction</h3><a id='link8'></a>

Modular image processing using [OpenCV](https://opencv.org/) and `Python` generators using [Detectron2](https://github.com/facebookresearch/detectron2). The idea of modularity allows the industrial processing pipeline to be programmed by several `DS` specialists. The idea of using generators is taken from [Arosław Gilewski](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696) - as a result, we get a full-fledged module that can be supplemented in the variations we need.

The aim of the work was to implement various approaches to integrating `Detectron2` and `OpenCV` into projects - both local and web using modules
or in the form of scripts using different versions of the models

> 📝 If you do not have a link to `medium` - turn on incognito mode

1. **Local execution**. In this version, you can work with random images, or use a trained model for detection
weapons, including video. By expanding the training data, you can get more accurate results.
The interface allows, without delving into the code, to perform image segmentation, including shading the background.

  Available functionality:

      - Images:
        - Instance segmentation on images
        - Key points
        - Panoptic segmentation
        - Separate background
        - Weapon detection (custom data set)
      - Video:
        - Instance segmentation on video
        - Show processing in real time
        - Weapon detection on video
        
  <p align="center"><img src="https://i.ibb.co/ZJSL1yL/2021-12-28-10-43-20.png" alt="2021-12-28-10-43-20" border="0"></p>

2. **Web Application**. An additional web application with limited functionality has been implemented (only images) - Located at: https://detartyseg.herokuapp.com/. Unfortunately, I had to cut the functionality due to the size restrictions for the application. Also, launching via [Docker](https://www.docker.com/) is possible - instructions in the corresponding part of the file [Readme.md](#linkdock) - locally, or after cloning the repository - launching via `app_local.py`

 Available functionality:

      - Images:
        - Instance segmentation on images
        - Keypoints on images
        - Panoptic segmentation (Deleted from web. Only in docker)
        - Separate background
        - Blur effect
        - Black and white image


<p align="center"><img src="https://i.ibb.co/mDCxTP3/2022-01-03-11-27-44.png" alt="2022-01-03-11-27-44" border="0"></p>


----
[Back to contest](#link7)


<h3>Local execution</h3><a id='link9'></a>
<h4> 1. Script process_img.py</h4><a id='link10'></a>

Here are three options for how the model works:

- `instance segmentation` into classes` COCO`
- `instance segmentation` for detecting weapons on a custom [dataset](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar). The images were taken from the [MLWhiz](https://github.com/MLWhiz/object_detection) repository, which I mapped out manually using [supervise.ly](https://supervise.ly/).

You can watch the training in this [notebook](https://nbviewer.org/github/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_detection_ipynb__.ipynb), if you want to train your detector (available in [Collab](https://colab.research.google.com/drive/1TOSFHYrQrxToQ4v5azP8xzpuLcgEKrQ3?usp=sharing)). The weights of the base models can be taken from the [zoo](https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py) - form `.yaml` similarly to those presented in the` config` folder
- `separate background` - separation of the background, instead of annotation. Idea suggested by [Arosław Gilewski](https://medium.com/deepvisionguru/how-to-embed-detectron2-in-your-computer-vision-project-817f29149461)
- `keypoints` - for building key points of people on images (` pose-flow`). 

For ease of launch, a simple `GUI` was added to this module, using [GOOEY](https://github.com/chriskiehl/Gooey)

---

**Install locally**

It is recommended to install to `virtuenv`. Optional step - can be skipped

```python
pip install virtualenv
virtualenv detection
source detection/bin/activate
```


It is also recommended to read the instructions to the end before starting the installation. For `Win` it is recommended to use` Ubuntu 20.04.3 LTS` with `WLS2` or` Conda` terminal. It is desirable that you already have [Anaconda](https://www.anaconda.com/products/individual)


Module configuration steps.

```python
git clone https://github.com/ArtyKrafty/detection
pip install -r requirements_loc.txt
```

Detectron2 must be installed in the adjacent directory with the module folder

<p align="center"><img src="https://i.ibb.co/R25y4Lx/2021-12-26-19-35-04.png" alt="2021-12-26-19-35-04" border="0"></p>

For Win, you must have Visual C ++ 2015 build tools installed
If not, install [from here](https://www.microsoft.com/en-ru/download/confirmation.aspx?id=48159) - this is required for [Detectron2](https://github.com/philferriere/cocoapi)


Also, it is recommended to check the version of torch and torchvision - you can check [here](https://pytorch.org/get-started/locally/) - select Package - copy the command to install


<p align="center"><img src="https://i.ibb.co/wc1NPMq/123.jpg" alt="123" border="0"></p>


> 📝 If you get an error - `Torchvision error: Could not find module image.pyd` - I recommend reading [here](https://kontext.tech/column/python/915/torchvision-error-could-not-find-module-imagepyd) with solution

For example, for `CPU`:

```python
pip3 install torch torchvision torchaudio
или
conda install pytorch torchvision cpuonly -c pytorch

```
`Detectron2` installation

```python
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

```python
Open terminal

cd detection
pythonw process_img.py

```

> 📝 Note that the call is made through `pythonw`, not` python` or `python3`

A window will open:

<p align="center"><img src="https://i.ibb.co/ZJSL1yL/2021-12-28-10-43-20.png" alt="2021-12-28-10-43-20" border="0"></p>


```python
You must set input. These are the images on which objects will be predicted.
You can use my photos - folder

./examples/images/photo/

also you can choose the confidence threshold.
Press START and in the outputs folder you have processed images
```
<p align="center"><img src="https://i.ibb.co/pz0Twxx/0299ce77-30e4-4b31-b8e5-fad85b140111.jpg" alt="0299ce77-30e4-4b31-b8e5-fad85b140111" border="0"></p>


2. `Instance segmentation` for detecting weapons on a custom [dataset](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar). The class names had to be neglected - otherwise they lost in universality. If you collect a larger set, you can use similar models to detect whether a person has a weapon

```python
Open terminal

cd detection
pythonw process_img.py

```

> 📝 Note that the call is made through `pythonw`, not` python` or `python3`

A window will open:

<p align="center"><img src="https://i.ibb.co/1TmBmfX/2021-12-28-10-10-42.png" alt="2021-12-28-10-10-42" border="0"></p>


```python
You must set input. These are the images on which objects will be predicted.
You can use my photos - folder

./examples/images/weapon/

also you can choose the confidence threshold. Here you need to change the model -
you will find the needed one in the config folder (and weights to it):


path ./detection/configs/COCO-InstanceSegmentation_weapon/mask_rcnn_R_50_FPN_3x.yaml.
We load weights to it: ./detection/configs/COCO-InstanceSegmentation_weapon/model_final.pth
Press START and in the outputs folder you have processed images
```
<p align="center"><img src="https://i.ibb.co/8M9Bt3T/new-use-hotel-robber-with-gun-2.jpg" alt="new-use-hotel-robber-with-gun-2" border="0"></p>

3. Separating the background - works the same way - just check the separate_background

```python
You must set input. These are the images on which objects will be predicted.
You can use my photos - folder

./examples/images/photos/

also you can choose the confidence threshold. Here you need to change the model -
you will find the needed one in the config folder (and weights to it):


path ./detection/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml.
We put a mark - separate background:
Press START and in the outputs folder you have processed images

```

Getting an image with a blur effect on the background

<p align="center"><img src="https://i.ibb.co/Q9m5nHh/004.jpg" alt="004" border="0"></p>

4. Building key points - works similarly to weapons - just specify the address to the configuration
```python
You must set input. These are the images on which objects will be predicted.
You can use my photos - folder

./examples/images/key_points/

also you can choose the confidence threshold. Here you need to change the model -
you will find the needed one in the config folder (and weights to it):


path ./detection/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml.
Press START and in the outputs folder you have processed images

There is no need to set weights.

```
We get an image with marked key points.


<p align="center"><img src="https://i.ibb.co/0KY5Jh1/predict.jpg" alt="predict" border="0"></p>

5. Panoptic group of images - just specify the address to the configuration

```python
You must set input. These are the images on which objects will be predicted.
You can use my photos - folder

./examples/images/pan/

also you can choose the confidence threshold. Here you need to change the model -
you will find the needed one in the config folder (and weights to it):


path ./detection/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml
Press START and in the outputs folder you have processed images

There is no need to set weights.

```

<p align="center"><img src="https://i.ibb.co/Zx9zB1g/predict.jpg" alt="predict" border="0"></p>

The goal was to demonstrate various options for the model without delving into the code on the part of the user of this module - it is enough to specify the address to the configuration

----
[Back to contest](#link7)

<h4> 2. Script process_video.py</h4><a id='link11'></a>

**Module usage**


`Instance segmentation` with `COCO`

```python
Open terminal

cd detection
pythonw process_video.py

```

> 📝 Note that the call is made through `pythonw`, not` python` or `python3`. The principle of operation is similar. Weapons can also be detected -
for ethical reasons - it was customary not to add the file with armed attacks to the repository

A window will open:

<p align="center"><img src="https://i.ibb.co/KKBhVRy/2021-12-27-23-26-40.png" alt="2021-12-27-23-26-40" border="0"></p>

```python
You must set input. This is a video in which objects will be predicted.
You can use my example

./examples/videos/cross_roads.mp4

also you can choose the confidence threshold.
Press START and in the outputs folder you have the processed video. Takes some time

```
> 📝 Important! At the moment, there is a [bug](https://github.com/facebookresearch/detectron2/issues/3780) for the class
`VideoVisualizer`. To get around it, the `Visualizer` class was used, which is used for images
therefore the masks change from frame to frame. Probably the bug is only observed on M1 silicone

![](sample.gif)
----
[Back to contest](#link7)

<h3>Web-application</h3><a id='link12'></a>
Script app_local.py - for local work


mirror for `app.py`, which is located at: https://detartyseg.herokuapp.com/.
Unfortunately - I had to cut the functionality, due to the size restrictions for the application. I had to remove the `panoptic segmentation` -
memory excess on the application, on a free server. Remained functional in the local application and `docker`

If you want to run locally, run `app_local.py`:


```python
 git clone https://github.com/ArtyKrafty/detection
 cd detection
 python app_local.py
 
```

An application with limited functionality will start at the address - `http: // YOUR_URL_from_terminal: 8080 /`


<p align="center"><img src="https://i.ibb.co/mDCxTP3/2022-01-03-11-27-44.png" alt="2022-01-03-11-27-44" border="0"></p>

Here you can load an image from the desktop, select masks on it or darken the background. It is also possible to download images by `URL` - only instance segmentation

<a id='linkdock'></a>

Another option is to run via Docker (local copy of https://detartyseg.herokuapp.com/) - weight of Image 14.4 GB:

```python
git clone https://github.com/ArtyKrafty/detection
cd detection

docker build . -f Dockerfile -t detectron2
docker run --name detectron2 -p 127.0.0.1:8080:8080 detectron2


and after finish:

docker stop detectron2
docker rm $(docker ps -a -q)

or you can use docker-desktop app
look files inside: docker run -t -i detectron2 /bin/bash
```

<p align="center"><img src="https://i.ibb.co/FwgVWJn/123.jpg" alt="123" border="0"></p>

> 📝 If you `MacOS` user - go to` docker -> advanced` settings and increase `RAM` from 2 to 4 GB - otherwise` C ++ `compiler
can't build `torch`. 


----
[Back to contest](#link7)

<h3>Help desk</h3><a id='linkprobeng'></a>

Problems do not arise with the application, but problems may arise when problems arise - `Detectron2`,` torch` and so on.
Here are some solutions to help you install if you choose to work outside of `docker`. Tested on `MacOs` and` Windows` - on `Win` - the main problem is the lack of` Microsoft Visual C++ `. This collection will help to solve the main problems when installing `torch` and` Detectron2`. Also, it is recommended to have Anaconda on board

| Problem  |  Description |  Solution |
|---|---|---|
| Web-app | Error `heroku` -` Application Error` appears | The error may be related to out of memory. You can try again to perform the action or take another picture. There will be no problems with local assembly |
| torch, Detectron | C ++ error on Windows when installing on `pycatools` | You do not have Visual Studio C ++ installed - the compiler is missing. Install [from here](https://www.microsoft.com/ru-ru/download/confirmation.aspx?id=48159) |
| torch, torchvision | outdated version or installed not matching your PC | you can check [here](https://pytorch.org/get-started/locally/) |
| torch | `Torchvision error: Could not find module image.pyd` | read [here](https://kontext.tech/column/python/915/torchvision-error-could-not-find-module-imagepyd) with a solution |
| Docker | error with `C ++` on assembly or `os` | go to `docker -> advanced` settings and increase the` RAM` memory from 2 GB to 4 GB |

----
[Back to contest](#link7)


<h3>Credits</h3><a id='linkeng001'></a>

1. [Deep learning school](https://www.dlschool.org/pro-track) - educate and guide.  
2. [Arosław Gilewski](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696) - CV engineer. Idea of using pipelines and separate background baseline.   
3. [Detectron2](https://github.com/facebookresearch/detectron2) - Meta library for working with images  
4. [OpenCV](https://opencv.org/) - video and image processing tool  
5. [Heroku](https://www.heroku.com/) - a fast way to develop and host applications  
6. [pyTorch](https://pytorch.org/docs/stable/torch.html) - framework for working with neural networks  
7. [Docker](https://www.docker.com/) - packaging and image transfer tool
8. [MLWhiz](https://github.com/MLWhiz/object_detection) - provided sample images for markup


----
[Back to contest](#link7)
