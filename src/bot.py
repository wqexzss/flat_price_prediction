import os
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from dotenv import load_dotenv
from telegram import ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)


BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=BASE_DIR / '.env')
BOT_TOKEN = os.getenv('BOT_TOKEN')

FEATURES = [
    'city',
    'area',
    'rooms',
    'floor',
    'total_floors',
    'distance_from_center',
    'is_first_floor',
    'is_last_floor',
    'floor_ratio',
]

CITY, AREA, ROOMS, FLOOR, TOTAL_FLOORS, DISTANCE = range(6)

model = CatBoostRegressor()
low_model = CatBoostRegressor()
high_model = CatBoostRegressor()

model.load_model(BASE_DIR / 'models' / 'catboost_model.cbm')
low_model.load_model(BASE_DIR / 'models' / 'catboost_low.cbm')
high_model.load_model(BASE_DIR / 'models' / 'catboost_high.cbm')


def predict_price(flat):
    flat = dict(flat)
    flat['is_first_floor'] = int(flat['floor'] == 1)
    flat['is_last_floor'] = int(flat['floor'] == flat['total_floors'])
    flat['floor_ratio'] = round(flat['floor'] / flat['total_floors'], 3)

    data = pd.DataFrame([flat])[FEATURES]

    price = model.predict(data)[0]
    low = low_model.predict(data)[0]
    high = high_model.predict(data)[0]

    return round(price), round(min(low, high)), round(max(low, high))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        'Привет! Я помогу примерно оценить квартиру.\n\n'
        'В каком городе находится квартира? Например: Москва'
    )
    return CITY


async def get_city(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['city'] = update.message.text.strip()
    await update.message.reply_text('Какая площадь квартиры? Например: 60')
    return AREA


async def get_area(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data['area'] = float(update.message.text.replace(',', '.'))
        await update.message.reply_text('Сколько комнат? Например: 2')
        return ROOMS
    except ValueError:
        await update.message.reply_text('Введи число, например: 60')
        return AREA


async def get_rooms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data['rooms'] = int(update.message.text)
        await update.message.reply_text('На каком этаже квартира? Например: 7')
        return FLOOR
    except ValueError:
        await update.message.reply_text('Введи целое число')
        return ROOMS


async def get_floor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data['floor'] = int(update.message.text)
        await update.message.reply_text('Сколько всего этажей в доме? Например: 16')
        return TOTAL_FLOORS
    except ValueError:
        await update.message.reply_text('Введи целое число')
        return FLOOR


async def get_total_floors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        total_floors = int(update.message.text)

        if total_floors < context.user_data['floor']:
            await update.message.reply_text('Всего этажей не может быть меньше этажа квартиры')
            return TOTAL_FLOORS

        context.user_data['total_floors'] = total_floors
        await update.message.reply_text(
            'Как далеко квартира от центра города? Введи расстояние в км, например: 8'
        )
        return DISTANCE
    except ValueError:
        await update.message.reply_text('Введи целое число')
        return TOTAL_FLOORS


async def get_distance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data['distance_from_center'] = float(update.message.text.replace(',', '.'))
        flat = dict(context.user_data)
        price, low, high = predict_price(flat)
        city = flat['city']
        area = flat['area']
        rooms = flat['rooms']
        floor = flat['floor']
        total_floors = flat['total_floors']
        distance = flat['distance_from_center']

        await update.message.reply_text(
            'Оценка квартиры\n\n'
            f'Город: {city}\n'
            f'Площадь: {area} кв. м\n'
            f'Комнат: {rooms}\n'
            f'Этаж: {floor} из {total_floors}\n'
            f'Расстояние от центра: {distance} км\n\n'
            f'Примерная цена: {price:,.0f}\n'
            f'Интервал: {low:,.0f} - {high:,.0f}',
            reply_markup=ReplyKeyboardRemove()
        )
        await update.message.reply_text('Чтобы оценить ещё одну квартиру, напиши /start')
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text('Введи число, например: 8')
        return DISTANCE
    except Exception as error:
        await update.message.reply_text(f'Ошибка: {error}')
        return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Отменено. Чтобы начать заново, напиши /start',
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


def main():
    if not BOT_TOKEN:
        raise ValueError('Не найден BOT_TOKEN в .env')

    app = Application.builder().token(BOT_TOKEN).build()
    conv = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_city)],
            AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_area)],
            ROOMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_rooms)],
            FLOOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_floor)],
            TOTAL_FLOORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_total_floors)],
            DISTANCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_distance)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    app.add_handler(conv)
    print('Бот запущен...')
    app.run_polling()


if __name__ == '__main__':
    main()
