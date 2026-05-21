import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ConversationHandler, ContextTypes
)
from catboost import CatBoostRegressor

BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=BASE_DIR / '.env')
BOT_TOKEN = os.getenv('BOT_TOKEN')

model = CatBoostRegressor()
low_model = CatBoostRegressor()
high_model = CatBoostRegressor()

model.load_model(BASE_DIR / 'models' / 'catboost_model.cbm')
low_model.load_model(BASE_DIR / 'models' / 'catboost_low.cbm')
high_model.load_model(BASE_DIR / 'models' / 'catboost_high.cbm')

FEATURES = [
    'area',
    'bedrooms',
    'bathrooms',
    'stories',
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'parking',
    'prefarea',
    'furnishingstatus',
]

(CITY, DISTANCE, AREA, BEDROOMS, BATHROOMS, STORIES,
 MAINROAD, GUESTROOM, BASEMENT,
 HOTWATER, AIRCON, PARKING, FURNISHING) = range(13)

YES_NO = [['Да', 'Нет']]
FURNISHING_KB = [['С мебелью'], ['Частично с мебелью'], ['Без мебели']]

YES_NO_VALUES = {
    'Да': 'yes',
    'Нет': 'no',
}

FURNISHING_VALUES = {
    'С мебелью': 'furnished',
    'Частично с мебелью': 'semi-furnished',
    'Без мебели': 'unfurnished',
}


def predict_price(flat):
    data = pd.DataFrame([flat])[FEATURES]

    price = model.predict(data)[0]
    low = low_model.predict(data)[0]
    high = high_model.predict(data)[0]

    return round(price), round(low), round(high)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        'Привет! Я оценю стоимость квартиры.\n\n'
        'В каком городе находится квартира?'
    )
    return CITY


async def get_city(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data['city'] = update.message.text.strip()
    await update.message.reply_text(
        'Как далеко квартира от центра города? Введи расстояние в км, например: 7'
    )
    return DISTANCE


async def get_distance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        distance = float(update.message.text.replace(',', '.'))
        ctx.user_data['distance_from_center'] = distance
        ctx.user_data['prefarea'] = 'yes' if distance <= 7 else 'no'
        await update.message.reply_text('Введи площадь, например: 5000')
        return AREA
    except ValueError:
        await update.message.reply_text('Введи число, например: 7')
        return DISTANCE


def get_yes_no_answer(text):
    return YES_NO_VALUES.get(text)


def get_furnishing_answer(text):
    return FURNISHING_VALUES.get(text)
    return AREA


async def get_area(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data['area'] = float(update.message.text.replace(',', '.'))
        await update.message.reply_text('Сколько спален? Например: 3')
        return BEDROOMS
    except ValueError:
        await update.message.reply_text('Введи число, например: 5000')
        return AREA


async def get_bedrooms(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data['bedrooms'] = int(update.message.text)
        await update.message.reply_text('Сколько ванных комнат? Например: 2')
        return BATHROOMS
    except ValueError:
        await update.message.reply_text('Введи целое число')
        return BEDROOMS


async def get_bathrooms(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data['bathrooms'] = int(update.message.text)
        await update.message.reply_text('Сколько этажей в доме? Например: 2')
        return STORIES
    except ValueError:
        await update.message.reply_text('Введи целое число')
        return BATHROOMS


async def get_stories(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data['stories'] = int(update.message.text)
        await update.message.reply_text(
            'Выход на главную дорогу?',
            reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
        )
        return MAINROAD
    except ValueError:
        await update.message.reply_text('Введи целое число')
        return STORIES


async def get_mainroad(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    answer = get_yes_no_answer(update.message.text)

    if not answer:
        await update.message.reply_text(
            'Выбери Да или Нет',
            reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
        )
        return MAINROAD

    ctx.user_data['mainroad'] = answer
    await update.message.reply_text(
        'Есть гостевая комната?',
        reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
    )
    return GUESTROOM


async def get_guestroom(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    answer = get_yes_no_answer(update.message.text)

    if not answer:
        await update.message.reply_text(
            'Выбери Да или Нет',
            reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
        )
        return GUESTROOM

    ctx.user_data['guestroom'] = answer
    await update.message.reply_text(
        'Есть подвал?',
        reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
    )
    return BASEMENT


async def get_basement(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    answer = get_yes_no_answer(update.message.text)

    if not answer:
        await update.message.reply_text(
            'Выбери Да или Нет',
            reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
        )
        return BASEMENT

    ctx.user_data['basement'] = answer
    await update.message.reply_text(
        'Есть горячее водоснабжение?',
        reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
    )
    return HOTWATER


async def get_hotwater(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    answer = get_yes_no_answer(update.message.text)

    if not answer:
        await update.message.reply_text(
            'Выбери Да или Нет',
            reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
        )
        return HOTWATER

    ctx.user_data['hotwaterheating'] = answer
    await update.message.reply_text(
        'Есть кондиционер?',
        reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
    )
    return AIRCON


async def get_aircon(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    answer = get_yes_no_answer(update.message.text)

    if not answer:
        await update.message.reply_text(
            'Выбери Да или Нет',
            reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
        )
        return AIRCON

    ctx.user_data['airconditioning'] = answer
    await update.message.reply_text(
        'Сколько парковочных мест?',
        reply_markup=ReplyKeyboardMarkup([['0', '1', '2', '3']], one_time_keyboard=True)
    )
    return PARKING


async def get_parking(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data['parking'] = int(update.message.text)
        await update.message.reply_text(
        'Состояние меблировки?',
        reply_markup=ReplyKeyboardMarkup(FURNISHING_KB, one_time_keyboard=True)
        )
        return FURNISHING
    except ValueError:
        await update.message.reply_text('Введи 0, 1, 2 или 3')
        return PARKING


async def get_furnishing(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    answer = get_furnishing_answer(update.message.text)

    if not answer:
        await update.message.reply_text(
            'Выбери вариант с клавиатуры',
            reply_markup=ReplyKeyboardMarkup(FURNISHING_KB, one_time_keyboard=True)
        )
        return FURNISHING

    ctx.user_data['furnishingstatus'] = answer
    flat = dict(ctx.user_data)

    try:
        price, low, high = predict_price(flat)
        city = flat['city']
        distance = flat['distance_from_center']

        await update.message.reply_text(
            'Оценка квартиры\n\n'
            f'Город: {city}\n'
            f'Расстояние от центра: {distance} км\n\n'
            f'Прогноз: {price:,.0f}\n'
            f'Интервал: {low:,.0f} - {high:,.0f}',
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as e:
        await update.message.reply_text(
            f'Ошибка: {e}',
            reply_markup=ReplyKeyboardRemove()
        )

    await update.message.reply_text('Хочешь оценить ещё? /start')
    return ConversationHandler.END


async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Отменено. /start - начать заново.',
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
            CITY:      [MessageHandler(filters.TEXT & ~filters.COMMAND, get_city)],
            DISTANCE:  [MessageHandler(filters.TEXT & ~filters.COMMAND, get_distance)],
            AREA:      [MessageHandler(filters.TEXT & ~filters.COMMAND, get_area)],
            BEDROOMS:  [MessageHandler(filters.TEXT & ~filters.COMMAND, get_bedrooms)],
            BATHROOMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_bathrooms)],
            STORIES:   [MessageHandler(filters.TEXT & ~filters.COMMAND, get_stories)],
            MAINROAD:  [MessageHandler(filters.TEXT & ~filters.COMMAND, get_mainroad)],
            GUESTROOM: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_guestroom)],
            BASEMENT:  [MessageHandler(filters.TEXT & ~filters.COMMAND, get_basement)],
            HOTWATER:  [MessageHandler(filters.TEXT & ~filters.COMMAND, get_hotwater)],
            AIRCON:    [MessageHandler(filters.TEXT & ~filters.COMMAND, get_aircon)],
            PARKING:   [MessageHandler(filters.TEXT & ~filters.COMMAND, get_parking)],
            FURNISHING: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_furnishing)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    app.add_handler(conv)
    print('Бот запущен...')
    app.run_polling()


if __name__ == '__main__':
    main()
