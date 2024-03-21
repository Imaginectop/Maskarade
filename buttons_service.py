from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def get_mask_choice_keyboard() -> InlineKeyboardMarkup:
    """
    Возвращает клавиатуру для выбора маски.

    :return: InlineKeyboardMarkup объект.
    """
    keyboard = [
        [InlineKeyboardButton("Маска 1", callback_data='mask_1')],
        [InlineKeyboardButton("Маска 2", callback_data='mask_2')],
        [InlineKeyboardButton("Маска 3", callback_data='mask_3')],

    ]
    return InlineKeyboardMarkup(keyboard)


def get_main_keyboard() -> InlineKeyboardMarkup:
    """
    Возвращает основную клавиатуру.

    :return: InlineKeyboardMarkup объект.
    """
    keyboard = [
        [InlineKeyboardButton("Добавить фото", callback_data='add_photo')],
        [InlineKeyboardButton("Отмена", callback_data='cancel')]
    ]
    return InlineKeyboardMarkup(keyboard)
