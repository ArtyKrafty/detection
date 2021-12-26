<p align="center"><img src='https://i.ibb.co/rbJmBSV/Computer-Vision-Object-Detection-original.jpg'></p>


Модульная обработка изображений с использованием [OpenCV](https://opencv.org/) and генераторов Python с помощью [Detectron2](https://github.com/facebookresearch/detectron2). Идея модульности позволяет программировать промышленный `pipeline` обработки несколькими специалистами DS. Идея использования генераторов взята у [Arosław Gilewski](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696) - в итоге получаем полноценный модуль, который может дополняться в нужных нам вариациях.


Здесь представлено два варианта работы модели:   

- `instance segmentation` на классы `COCO`
- `instance segmentation` для обнаружения оружия на кастомном [датасете](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar). Изображения были взяты с репозитория [MLWhiz](https://github.com/MLWhiz/object_detection), которые я разметил вручную с помощью [supervise.ly](https://supervise.ly/) 


Для удобства запуска, данному модулю было добавлено простое `GUI`, с помощью [GOOEY](https://github.com/chriskiehl/Gooey)

---
**Установка. Рекомендуется устанавливать в `virtuenv`**

```
pip install virtualenv
virtualenv ENV
source ENV/bin/activate

 Windows путь модет отличаться. Scripts вместо bins. н-р

```
Шаги настройки модуля

```
1. git clone https://github.com/ArtyKrafty/detection
2. pip install -r /path/to/requirements.txt - этот шаг можно проспутить, если у вас 
установлены torch, trochvision, cv2

Для работы необходимо, чтобы были установлены Detectron и wxPython

3. pip install -U wxPython
4. Detectron необходимо установить в соседний каталог с папкой модуля

```

<img src="https://i.ibb.co/R25y4Lx/2021-12-26-19-35-04.png" alt="2021-12-26-19-35-04" border="0">

```
cd ..
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python setup.py build develop

Если возникнут сложности, можно обратиться к официальной документации

```
[Официальное руководство по установке Detectron2](https://github.com/facebookresearch/detectron2/blob/3def12bdeaacd35c6f7b3b6c0097b7bc31f31ba4/INSTALL.md)


___

**Использование модуля**


1. `Instance segmentation` на классы `COCO`

```
Открываем терминал

cd detection
pythonw proces_img.py

```

Обратите внимание, что вызов осуществляется через `pythonw`, не `python` или `python3`

У вас откроется окно:

<img src="https://i.ibb.co/fxTtpK0/2021-12-26-19-41-55.png" alt="2021-12-26-19-41-55" border="0">

```
Необходимо задать input. Это изображения, на которых будут предсказываться объекты. 
Вы можете воспользоваться моими фотографиями - папка

./examples/images/photo/

также вы можете выбрать порог confidence. 
Нажимаем START и в папке outputs у вас обработанные изображения

```
<img src="https://i.ibb.co/pz0Twxx/0299ce77-30e4-4b31-b8e5-fad85b140111.jpg" alt="0299ce77-30e4-4b31-b8e5-fad85b140111" border="0">


2. `Instance segmentation` для обнаружения оружия на кастомном [датасете](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar)

```
Открываем терминал

cd detection
pythonw proces_img.py

```

Обратите внимание, что вызов осуществляется через `pythonw`, не `python` или `python3`

У вас откроется окно:

<img src="https://i.ibb.co/k9JXL3T/2021-12-26-19-50-35.png" alt="2021-12-26-19-50-35" border="0">

```
Необходимо задать input. Это изображения, на которых будут предсказываться объекты. 
Вы можете воспользоваться моими фотографиями - папка

./examples/images/weapon/

также вы можете выбрать порог confidence. Тут необходимо поменять модель - 
вы найдете нужную в папке config:   


путь ./detection/configs/COCO-InstanceSegmentation_weapon/mask_rcnn_R_50_FPN_3x.yaml. 
Грузим к ней веса: ./detection/configs/COCO-InstanceSegmentation_weapon/model_final.pth
Нажимаем START и в папке outputs у вас обработанные изображения
```
<img src="https://i.ibb.co/8M9Bt3T/new-use-hotel-robber-with-gun-2.jpg" alt="new-use-hotel-robber-with-gun-2" border="0">




