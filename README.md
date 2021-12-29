<p align="center"><img src='https://i.ibb.co/rbJmBSV/Computer-Vision-Object-Detection-original.jpg'></p>


Модульная обработка изображений с использованием [OpenCV](https://opencv.org/) и генераторов Python с помощью [Detectron2](https://github.com/facebookresearch/detectron2). Идея модульности позволяет программировать промышленный `pipeline` обработки несколькими специалистами DS. Идея использования генераторов взята у [Arosław Gilewski](https://medium.com/deepvisionguru/modular-image-processing-pipeline-using-opencv-and-python-generators-9edca3ccb696) - в итоге получаем полноценный модуль, который может дополняться в нужных нам вариациях.  

*Примечание* Если у Вас не открывается ссылка на medium - включите режим инкогнито

В данном исполнении - вы можете работать со случаными изображениями, либо применять обученную модель для детекции
оружия, в том числе на видео. При расширении обучающих данных, можно получить более точные результаты.
Интерфейс позволяет, не углубляясь в код, производить сегментацию изображений, в том числе и затенять фон

<h3> 1. Скрипт process_img.py</h3>

Здесь представлено два варианта работы модели:   

- `instance segmentation` на классы `COCO`
- `instance segmentation` для обнаружения оружия на кастомном [датасете](https://github.com/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_data.tar). Изображения были взяты с репозитория [MLWhiz](https://github.com/MLWhiz/object_detection), которые я разметил вручную с помощью [supervise.ly](https://supervise.ly/). Обучение вы можете посмотреть в этом [ноутбуке](https://nbviewer.org/github/ArtyKrafty/detection/blob/main/configs/COCO-InstanceSegmentation_weapon/weapon_detection_ipynb__.ipynb), если захотите обучить свой детектор (доступно и в [Collab](https://colab.research.google.com/drive/1TOSFHYrQrxToQ4v5azP8xzpuLcgEKrQ3?usp=sharing)). Веса базовых моделей можно брать из [зоопарка](https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py) - формировать `.yaml` аналогично представленным в папке `config`
- `separate background` - отделение фона, вместо аннотации. Идея предложена [Arosław Gilewski](https://medium.com/deepvisionguru/how-to-embed-detectron2-in-your-computer-vision-project-817f29149461)


Для удобства запуска, данному модулю было добавлено простое `GUI`, с помощью [GOOEY](https://github.com/chriskiehl/Gooey)

---
**Установка локально**  

Рекомендуется устанавливать в `virtuenv`.
Также рекомендуется дочитать инструкции до конца перед началом установки

```

pip install virtualenv
virtualenv detection
source detection/bin/activate

 Windows путь модет отличаться. Scripts вместо bins. н-р
 Не используйте conda если у вас M1 silicone - detectron2 тогда может 
 работать некорректно и не будет установлен

```
Шаги настройки модуля

```
git clone https://github.com/ArtyKrafty/detection
pip install -r requirements.txt - если требуется. Шаг можно пропустить,  
установить требуемые библиотеки вручную

Для работы необходимо, чтобы были установлены Detectron, wxPython  - важно установить. 
Необходимо для работы GUI, torch, trochvision, cv2

pip install -U wxPython


Detectron2 необходимо установить в соседний каталог с папкой модуля



```

<img src="https://i.ibb.co/R25y4Lx/2021-12-26-19-35-04.png" alt="2021-12-26-19-35-04" border="0">

```
cd

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# На macOS, необходимо установить так (особенно, если у вас M1)
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install -e detectron2

Если возникнут сложности, можно обратиться к официальной документации

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


<h3> 2. Скрипт process_video.py</h3>

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

3. Скрипт app.py

Находится по адресу: 

Если хотите запустить локально  - необходимо в файле app.py поменять строку: `app.run(debug=True)` на следующие (уже в прекоде):
```

```    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)

```
 cd detection
 python app.py
 
```
Запустится приложение с ограниченным функционалом по адресу - `http://ВАШ_URL_из_терминала:8080/`


<img src="https://i.ibb.co/nzYcSwQ/2021-12-29-14-34-18.png" alt="2021-12-29-14-34-18" border="0">

Здесь вы можете загрузить изображение с рабочего стола, выделить маски на нем или затемнить фон. Также возможно загрузка изображений по `URL` - только instance segmentation
