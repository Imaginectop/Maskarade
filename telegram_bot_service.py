import logging
from telegram import Update, InputMediaPhoto
from telegram.ext import Updater, CommandHandler, Filters, MessageHandler, CallbackQueryHandler, CallbackContext
from config import TELEGRAM_TOKEN
from image_processing_service import apply_mask
from mask_management_service import get_mask
from buttons_service import get_main_keyboard, get_mask_choice_keyboard

# Инициализация логгирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Прикрепите и отправьте мне фотографию, и я наложу на нее маску.', reply_markup=get_main_keyboard())


def handle_photo(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    photo_file = update.message.photo[-1].get_file()
    photo_path = f"photo_{user_id}.jpg"
    photo_file.download(photo_path)
    
    context.user_data['photo_path'] = photo_path  # Сохраняем путь к фото в context.user_data для дальнейшего использования
    update.message.reply_text("Выберите маску:", reply_markup=get_mask_choice_keyboard())


def handle_button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()

    if query.data in ["mask_1", "mask_2", "mask_3","1LMaskBlue","2LMaskBlue","3LMaskBlue","4Al;LMaskBlue","4LMaskBlue","5LMaskBlue"]:
        mask_path = get_mask(query.data)  # Получаем путь к маске (можно модифицировать, чтобы выбирать конкретную маску в зависимости от callback_data)
        photo_path = context.user_data.get('photo_path')  # Извлекаем путь к фото из context.user_data

        if not photo_path:
            query.edit_message_text("Ошибка обработки фото. Пожалуйста, попробуйте снова.", reply_markup=get_main_keyboard())
            return

        result_path, success = apply_mask(photo_path, mask_path)
        if success:
            with open(result_path, 'rb') as photo:
                query.edit_message_text("Ваше фото готово!")
                query.message.reply_photo(photo=photo)
        else:
            query.edit_message_text("Не удалось наложить маску. Пожалуйста, попробуйте снова.", reply_markup=get_main_keyboard())
    elif query.data == "add_photo":
        # Отправка образца фото
        with open("sample_photo/sample.jpg", 'rb') as sample_photo:
            instructions = ("Пожалуйста, отправьте фото для обработки."
                            "\nИнструкции по загрузке фото:"
                            "\n1. Убедитесь, что лицо хорошо освещено и находится в центре фото."
                            "\n2. Избегайте использования фильтров."
                            "\n3. Убедитесь, что лицо полностью видно на фото."
                            "\n4. Для повторного использования введите /start."
                            "\nВот образец правильного фото:")
            query.message.reply_photo(photo=sample_photo, caption=instructions)

    elif query.data == "cancel":
        query.edit_message_text("Действие отменено. Что бы вы хотели сделать дальше?", reply_markup=get_main_keyboard())


def main() -> None:
    updater = Updater(TELEGRAM_TOKEN)

    # Получаем диспетчер для регистрации обработчиков
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))
    dp.add_handler(CallbackQueryHandler(handle_button))

    # Запускаем бота
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
