import os
import shutil
import subprocess
import tempfile
import configparser

from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from ultralytics import YOLO


# Создаем директорию tmpfs
os.makedirs('./mnt/my_tmpfs', exist_ok=True)
# Монтируем tmpfs
subprocess.run(["sudo", "mount", "-t", "tmpfs", "-o", "size=512m", "tmpfs", "./mnt/my_tmpfs"], check=True)

# Парсим конфигурационный файл
config = configparser.ConfigParser()
config.read('bot_config.ini')

# Загружаем переменные окружения
load_dotenv()

# Устанавливаем токен бота
TOKEN = os.environ.get("TG_TOKEN")

# инициализируем класс YOLO
model = YOLO(config['YOLO']['MODEL_PATH'])

# функция команды /start
async def start(update, context):
    await update.message.reply_text('🤖 Пришлите видео для трекинга людей!')

# функция обработки видео
async def tracking(update, context):

    # удаляем папку runs с результатом предыдущего распознавания
    shutil.rmtree('./mnt/my_tmpfs/runs', ignore_errors=True)

    # текстовое сообщение пользователю
    delete_message = await update.message.reply_text('🎦 Видео получено.\nИдёт обработка... ♻️')

    # получение файла из сообщения
    new_file = await update.message.video.get_file()

    # имя файла в tmpfs
    video_name = str(new_file.file_path).split("/")[-1]

    # создаем файл в оперативной памяти
    tmp_file = tempfile.NamedTemporaryFile(delete=True, dir="./mnt/my_tmpfs")
    tmp_filename = tmp_file.name

    # скачиваем файл во временный файл
    new_file.download(custom_path=tmp_filename)

    # ширина и высота кадров
    height = update.message.video.height
    width = update.message.video.width

    # проверяем формат видео
    if video_name.endswith('MOV'):
        # конвертация видео из формата .mov в .mp4
        video = VideoFileClip(tmp_filename)
        video.write_videofile(f"{tmp_filename[:-4]}.mp4")

    # распознавание объектов
    # experiment_name = "experiment" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')  # имя подпапки эксперимента
    model.track(source=f"{tmp_filename[:-4]}.mp4",  # путь к файлу
                conf=0.35,             # порог уверенности
                iou=0.5,               # intersection over union
                imgsz=(640, 640),      # размер кадра
                show=False,            # показывать ли результат
                save=True,             # сохранять ли результат
                classes=[0, 1, 2, 9],  # классы для распознавания
                save_txt=False,        # сохранение разметки распознанных объектов
                save_conf=False)       # сохранение значений уверенности объектов
                # project='my_dir',     # путь проекта для сохранения
                #name=experiment_name) # имя папки для сохранения внутри проекта

    # конвертация видео из формата .mp4 в .mov
    #video = VideoFileClip(f"./mnt/my_tmpfs/runs/detect/track/{tmp_filename[:-4]}.mp4")
    #video.write_videofile(f"./mnt/my_tmpfs/runs/detect/track/{tmp_filename[:-4]}_mov.mov", codec='libx264')

    # удаляем предыдущее сообщение от бота
    await context.bot.deleteMessage(chat_id=update.message.chat_id, message_id=delete_message.message_id)

    # отправляем пользователю результат
    await update.message.reply_text('✅ Распознавание объектов завершено ❇️')
    await update.message.reply_video(f"./mnt/my_tmpfs/runs/detect/track/{tmp_filename[:-4]}_mov.mov", width=width, height=height)


def main():
    # точка входа в приложение .write_timeout(30)
    application = Application.builder().token(TOKEN).read_timeout(100).build()
    print('🔘 Бот запущен...')

    # добавляем обработчик команды /start
    application.add_handler(CommandHandler("start", start))

    # добавляем обработчик видеозаписей
    application.add_handler(MessageHandler(filters.VIDEO, tracking))

    # запуск приложения (для остановки нужно нажать Ctrl-C)
    application.run_polling()

if __name__ == "__main__":
    main()
