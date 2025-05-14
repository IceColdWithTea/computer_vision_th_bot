import asyncio
from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton
)
from ultralytics import YOLO
import os
import cv2




BOT_TOKEN = ''
router = Router()

model = YOLO("yolov8n.pt")

# Выбранные классы для нашего бота с эмодзи
SELECTED_CLASSES = {
    "car": "🚗 Машины",
    "person": "🧍 Люди",
    "dog": "🐕 Собаки",
    "cat": "🐈 Кошки",
    "bus": "🚌 Автобусы",
    "cake": "🍰 Пирожные"
}



CLASS_COLORS = {
    "car": (0, 255, 0),
    "person": (255, 0, 0),
    "dog": (0, 0, 255),
    "cat": (255, 255, 0),
    "bus": (0, 165, 255),
    "cake": (255, 0, 255)
}

user_choices = {}


def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📋 Меню")]],
        resize_keyboard=True,
        input_field_placeholder="Нажмите Меню для выбора объектов"
    )


def get_class_selection_keyboard(user_id: int):
    if user_id not in user_choices:
        user_choices[user_id] = {class_id: False for class_id in SELECTED_CLASSES}

    buttons = []
    for class_id, class_name in SELECTED_CLASSES.items():
        selected = "✅ " if user_choices[user_id][class_id] else ""
        buttons.append([InlineKeyboardButton(
            text=f"{selected}{class_name}",
            callback_data=f"select_{class_id}"
        )])

    buttons.append([
        InlineKeyboardButton(text="🔍 Анализировать фото", callback_data="analyze_photo")
    ])

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def draw_boxes(image, results, selected_classes):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]

            if class_name in selected_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = CLASS_COLORS.get(class_name, (0, 255, 255))

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                confidence = float(box.conf[0])
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@router.message(Command("start"))
async def start(message: types.Message):
    user_id = message.from_user.id
    user_choices[user_id] = {class_id: False for class_id in SELECTED_CLASSES}
    await message.answer(
        "Я бот для анализа изображений. Нажмите 'Меню' для выбора объектов.",
        reply_markup=get_main_keyboard()
    )


@router.message(lambda message: message.text == "📋 Меню")
async def show_menu(message: types.Message):
    await message.answer(
        "Выберите объекты для поиска, нажмите анализировать фото, затем отправьте само фото:",
        reply_markup=get_class_selection_keyboard(message.from_user.id)
    )


@router.callback_query(lambda c: c.data.startswith("select_"))
async def toggle_class_selection(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    class_id = callback.data.split("_")[1]

    if user_id not in user_choices:
        user_choices[user_id] = {class_id: False for class_id in SELECTED_CLASSES}

    user_choices[user_id][class_id] = not user_choices[user_id][class_id]

    try:
        await callback.message.edit_reply_markup(
            reply_markup=get_class_selection_keyboard(user_id)
        )
    except Exception as e:
        if "message is not modified" not in str(e):
            raise e
    await callback.answer()


@router.callback_query(lambda c: c.data == "analyze_photo")
async def request_photo_analysis(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    if not any(user_choices.get(user_id, {}).values()):
        await callback.answer("❌ Выберите хотя бы один объект!", show_alert=True)
        return

    await callback.message.answer(
        "📷 Отправьте фото для анализа",
        reply_markup=get_main_keyboard()
    )
    await callback.answer()


@router.message(lambda message: message.photo)
async def analyze_photo(message: types.Message, bot: Bot):
    user_id = message.from_user.id
    if user_id not in user_choices or not any(user_choices[user_id].values()):
        await message.answer(
            "❌ Сначала выберите объекты в меню!",
            reply_markup=get_main_keyboard()
        )
        return

    # Скачиваем фото
    file_id = message.photo[-1].file_id
    file = await bot.get_file(file_id)
    input_image = f"temp_{file_id}.jpg"
    await bot.download_file(file.file_path, input_image)

    try:
        # Читаем изображение
        image = cv2.imread(input_image)
        if image is None:
            raise ValueError("Не удалось загрузить изображение")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Получаем ID выбранных классов
        selected_classes = [k for k, v in user_choices[user_id].items() if v]
        class_ids = [i for i, name in model.names.items() if name in selected_classes]

        if not class_ids:
            await message.answer("❌ Не удалось определить выбранные классы")
            return

        # Обработка фото через YOLO
        results = model.predict(
            source=image,
            classes=class_ids,
            conf=0.5
        )

        counts = {class_name: 0 for class_name in selected_classes}
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls)]
                if class_name in counts:
                    counts[class_name] += 1

        image_with_boxes = draw_boxes(image, results, selected_classes)

        output_image = f"result_{file_id}.jpg"
        cv2.imwrite(output_image, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))

        result_text = "🔍 Результаты анализа:\n"
        for class_name, count in counts.items():
            result_text += f"{SELECTED_CLASSES.get(class_name, class_name)}: {count}\n"

        with open(output_image, "rb") as photo_file:
            await message.answer_photo(
                types.BufferedInputFile(
                    photo_file.read(),
                    filename="result.jpg"
                ),
                caption=result_text,
                reply_markup=get_main_keyboard()
            )

    except Exception as e:
        await message.answer(
            f"⚠️ Ошибка обработки: {str(e)}",
            reply_markup=get_main_keyboard()
        )
    finally:
        if os.path.exists(input_image):
            os.remove(input_image)
        if 'output_image' in locals() and os.path.exists(output_image):
            os.remove(output_image)


async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    print("Bot started")
    asyncio.run(main())