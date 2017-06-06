#!/usr/bin/python3
import cv2
import numpy as np


# Чтение данных из файла
def read_csv(filename, images, labels):
    with open(filename) as file:
        for line in file:
            image, label = line.split(';')
            images.append(cv2.imread(image, 0))
            labels.append(int(label))

images_list = []
labels_list = []
# Создание объекта для работы с камерой
cap = cv2.VideoCapture(0)
# Установка разрешения 320*240
cap.set(3, 320)
cap.set(4, 240)
# Создание классификатора и загрузка его параметров из файла
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
read_csv('faces.csv', images_list, labels_list)
# Сохранение размеров картинок из выборки
im_height, im_width = images_list[0].shape
# Преобразование массива в массив numpy
labels_list = np.array(labels_list)
# Создание модели распознователя лиц
model = cv2.face.createFisherFaceRecognizer()
# Обучение модели
model.train(images_list, labels_list)

while True:
    # Чтение кадра с камеры
    ret, frame = cap.read()
    # Если кадр пустой, то выходим из программы
    if not ret:
        break
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Поиск лиц на изображении
    faces = face_cascade.detectMultiScale(gray, 1.15, 3)
    # Цикл по всем лицам
    for (x, y, width, height) in faces:
        # Вырезаем область лица из чёрно-белой картинки
        face = gray[y:y + height, x:x + width]
        # Масштабируем его до размера лиц из выборки
        face = cv2.resize(face, (im_height, im_width))
        # Определяем чьё это лицо
        prediction, confidence = model.predict(face)
        text = "Prediction = "
        if prediction == 0:
            text += "Kostya"
        else:
            text += "Nastya"
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame,  # изображение
                      (x, y),  # координаты левого верхнего угла
                      (x + width, y + height),  # ширина и высота
                      (255, 0, 0),  # Цвет
                      3)  # Толщина линии
        # Добавляем текст с именем
        cv2.putText(frame, text, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 2)

    # Показываем изображение
    cv2.imshow("Frame", frame)

    # При нажатии клавиши Esc - выход из программы
    if cv2.waitKey(1) == 27:
        break
