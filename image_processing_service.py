import cv2
import os
import numpy as np
from face_detection_service import detect_faces
import logging

TEMP_FILES_DIR = "temp_files"

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_mask(image_path: str, mask_path: str) -> (str, bool):
    if not os.path.exists(image_path):
        logger.error(f"Файл изображения {image_path} не найден.")
        return None, False
    
    if not os.path.exists(mask_path):
        logger.error(f"Файл маски {mask_path} не найден.")
        return None, False

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Файл изображения {image_path} не может быть прочитан.")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Файл маски {mask_path} не может быть прочитан.")
    except FileNotFoundError as e:
        logger.error(f"Ошибка при загрузке файла: {e}")
        return None, False
    except Exception as e:
        logger.error(f"Неизвестная ошибка при чтении файла: {e}")
        return None, False

    faces = detect_faces(image_path)

    if not faces:
        return image_path, False

    (startX, startY, endX, endY) = faces[0]
    face_width = endX - startX
    face_height = endY - startY

    new_width = int(face_width * 1.1)
    new_height = int(face_height * 1.1)

    try:
        mask_resized = cv2.resize(mask, (new_width, new_height))
    except Exception as e:
        logger.error(f"Ошибка при изменении размера маски: {e}")
        return None, False

    dx = (new_width - face_width) // 2
    dy = (new_height - face_height) // 2

    for i in range(mask_resized.shape[0]):
        for j in range(mask_resized.shape[1]):
            if mask_resized[i, j, 3] > 0:
                if 0 <= startY + i - dy < image.shape[0] and 0 <= startX + j - dx < image.shape[1]:
                    image[startY + i - dy, startX + j - dx, :] = mask_resized[i, j, :-1]

    result_path = os.path.join(TEMP_FILES_DIR, f"masked_{os.path.basename(image_path)}")
    cv2.imwrite(result_path, image)

    return result_path, True
