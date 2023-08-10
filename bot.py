from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from io import BytesIO
import cv2
import os
import numpy as np
import torch
import random
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ.get("TOKEN")
# раскомментировать если при запуске скрипта появляется ошибка OMP
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov8n.pt')
imgsz = 640

# Определение цветов для различных классов (используем случайные цвета для визуализации)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(1000)]

async def start(update, context):
    await update.message.reply_text('🪬 Пришлите видео для трекинга людей!')

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def detect_object(img):
    pred = model(img)[0]
    pred = non_max_suppression(pred)
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape).round()
    return pred

# функция обработки видео
async def tracking(update, context):
    video_file = await context.bot.get_file(update.message.video.file_id)

    delete_message = await update.message.reply_text('Видео получено. Идёт обработка...')

    file_path = "temp_video_file.mp4"
    video_file.download(custom_path=file_path)

    video = cv2.VideoCapture(file_path)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        img = letterbox(frame, imgsz)[0]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        detections = detect_object(img)
        
        # Обработка обнаружений
        for i, det in enumerate(detections): 
            if len(det):  
                for *xyxy, conf, cls in reversed(det):
                    label = f'{conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls) % len(colors)], line_thickness=3)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Нажмите q для выхода
            break
    video.release()

    # удаляем предыдущее сообщение от бота
    await context.bot.delete_message(message_id = delete_message.message_id,
                                    chat_id = update.message.chat_id)
    
    # отправляем пользователю результат
    await update.message.reply_text('Распознавание объектов завершено')
    # Здесь вы должны сохранить обработанный видеофайл и затем отправить его
    # await update.message.reply_video(video)

def main():  # точка входа в приложение .write_timeout(30)
    application = Application.builder().token(TOKEN).read_timeout(100).build()
    print('Бот запущен...')

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VIDEO, tracking))
    application.run_polling()

if __name__ == "__main__":
    main()
