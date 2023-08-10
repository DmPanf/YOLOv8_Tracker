# yolo.py
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import os
import datetime

# раскомментировать если при запуске скрипта появляется ошибка OMP
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# путь к видеофайлу
path_to_file = 'video/UrbanFootball.mp4'

def main():

    # инициализируем класс YOLO
    model = YOLO('yolov8n.pt')

    # распознавание объектов
    experiment_name = "experiment" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')  # имя подпапки эксперимента
    model.track(source=path_to_file,  # путь к файлу
                conf=0.35,            # порог уверенности
                iou=0.5,              # intersection over union
                imgsz=(640, 640),     # размер кадра
                show=False,           # показывать ли результат
                save=True,            # сохранять ли результат
                classes=[0, 1, 2, 9], # классы для распознавания
                save_txt=False,       # сохранение разметки распознанных объектов
                save_conf=False,      # сохранение значений уверенности объектов
                project='my_dir',     # путь проекта для сохранения
                name=experiment_name) # имя папки для сохранения внутри проекта


if __name__ == "__main__":
    main()
