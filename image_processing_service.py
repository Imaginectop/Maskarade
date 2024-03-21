import cv2
import os
import numpy as np
from face_detection_service import detect_faces

TEMP_FILES_DIR = "temp_files"


def apply_mask(image_path: str, mask_path: str) -> (str, bool):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    faces = detect_faces(image_path)

    # Если лица не обнаружены, возвращаем путь к исходному изображению и False
    if not faces:
        return image_path, False

    # В данной реализации мы применяем маску только к первому обнаруженному лицу
    (startX, startY, endX, endY) = faces[0]
    face_width = endX - startX
    face_height = endY - startY

    # Увеличиваем размер маски на 10%
    new_width = int(face_width * 1.1)
    new_height = int(face_height * 1.1)

    # Масштабируем маску
    mask_resized = cv2.resize(mask, (new_width, new_height))

    # Центрирование маски относительно лица
    dx = (new_width - face_width) // 2
    dy = (new_height - face_height) // 2

    # Накладываем маску на изображение
    for i in range(mask_resized.shape[0]):
        for j in range(mask_resized.shape[1]):
            if mask_resized[i, j, 3] > 0:  # Если маска не прозрачна
                # Проверка границ перед присваиванием
                if 0 <= startY + i - dy < image.shape[0] and 0 <= startX + j - dx < image.shape[1]:
                    image[startY + i - dy, startX + j - dx, :] = mask_resized[i, j, :-1]

    result_path = os.path.join(TEMP_FILES_DIR, f"masked_{os.path.basename(image_path)}")
    cv2.imwrite(result_path, image)

    return result_path, True
