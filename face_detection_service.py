import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Загрузка модели при инициализации модуля
net = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, "deploy.prototxt"),
    os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
)


def detect_faces(image_path: str) -> list:
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Препроцессинг изображения
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # порог уверенности
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
    
    return faces
