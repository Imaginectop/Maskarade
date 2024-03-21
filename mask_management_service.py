import os

MASKS_DIR = 'masks'


def get_mask(mask_name: str) -> str:
    mask_path = os.path.join(MASKS_DIR, f"{mask_name}.png")
    if os.path.exists(mask_path):
        return mask_path
    else:
        raise ValueError(f"Маска с именем {mask_name} не найдена!")
