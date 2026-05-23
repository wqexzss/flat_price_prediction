import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from dotenv import load_dotenv
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=BASE_DIR / '.env')
BOT_TOKEN = os.getenv('BOT_TOKEN', '').strip()

MODEL_DIR = BASE_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'input_catboost_model.cbm'
MODEL_LOW_PATH = MODEL_DIR / 'input_catboost_low.cbm'
MODEL_HIGH_PATH = MODEL_DIR / 'input_catboost_high.cbm'
CONFIG_PATH = MODEL_DIR / 'input_model_config.json'
LOCK_PATH = BASE_DIR / '.bot.lock'

REGION, AREA, ROOMS, FLOOR, TOTAL_FLOORS, KITCHEN, BUILDING_TYPE, OBJECT_TYPE, POSTAL_CODE, COORDS = range(10)

REGION_CODES = {
    'Москва': '77',
    'Санкт-Петербург': '78',
    'Московская область': '50',
    'Краснодарский край': '23',
    'Свердловская область': '66',
    'Татарстан': '16',
    'Новосибирская область': '54',
    'Ленинградская область': '47',
}

BUILDING_TYPES = {
    '0': 'не знаю',
    '1': 'панельный',
    '2': 'монолитный',
    '3': 'кирпичный',
    '4': 'блочный',
    '5': 'деревянный',
    '6': 'другой',
}

OBJECT_TYPES = {
    '0': 'вторичка',
    '2': 'новостройка',
}


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            'Не найден models/input_model_config.json. '
            'Сначала запусти обучение в src/model.ipynb.'
        )

    with open(CONFIG_PATH, encoding='utf-8') as file:
        return json.load(file)


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(f'Не найдена модель {path}. Сначала обучи модель.')

    loaded_model = CatBoostRegressor()
    loaded_model.load_model(path)
    return loaded_model


config = load_config()
FEATURES = config['features']
CAT_FEATURES = set(config.get('cat_features', []))
NUMERIC_DEFAULTS = config.get('numeric_medians', {})
CAT_DEFAULTS = config.get('cat_defaults', {})
REGION_DEFAULTS = config.get('region_defaults', {})

model = load_model(MODEL_PATH)
model_low = load_model(MODEL_LOW_PATH)
model_high = load_model(MODEL_HIGH_PATH)
lock_file = None


def lock_single_instance():
    global lock_file

    if os.name != 'posix':
        return

    import fcntl

    lock_file = open(LOCK_PATH, 'w', encoding='utf-8')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error('Бот уже запущен. Останови старый процесс или терминал с bot.py.')
        sys.exit(1)

    lock_file.write(str(os.getpid()))
    lock_file.flush()


def safe_divide(first, second, default=0):
    if second in (0, None) or pd.isna(second):
        return default
    return first / second


def make_features(flat):
    # сначала ставим обычные значения из обучения
    row = {}
    for feature in FEATURES:
        if feature in CAT_FEATURES:
            row[feature] = CAT_DEFAULTS.get(feature, '-1')
        else:
            row[feature] = NUMERIC_DEFAULTS.get(feature, 0)

    area = float(flat['area'])
    kitchen_area = float(flat['kitchen_area'])
    level = int(flat['level'])
    levels = int(flat['levels'])
    rooms = int(flat['rooms'])
    id_region = str(flat['id_region'])

    region_default = REGION_DEFAULTS.get(id_region, {})
    geo_lat = float(flat.get('geo_lat') or region_default.get('geo_lat') or row.get('geo_lat', 55.75))
    geo_lon = float(flat.get('geo_lon') or region_default.get('geo_lon') or row.get('geo_lon', 37.62))
    postal_code = str(flat.get('postal_code') or region_default.get('postal_code') or CAT_DEFAULTS.get('postal_code', '-1'))

    row.update({
        'area': area,
        'kitchen_area': kitchen_area,
        'level': level,
        'levels': levels,
        'rooms': rooms,
        'geo_lat': geo_lat,
        'geo_lon': geo_lon,
        'building_type': str(flat['building_type']),
        'object_type': str(flat['object_type']),
        'id_region': id_region,
        'postal_code': postal_code,
    })

    # тут делаем такие же признаки как в ноутбуке
    row['floor_ratio'] = safe_divide(level, levels)
    row['is_first_floor'] = int(level == 1)
    row['is_last_floor'] = int(level == levels)
    row['area_per_room'] = safe_divide(area, rooms if rooms > 0 else 1)
    row['kitchen_ratio'] = safe_divide(kitchen_area, area)
    row['log_area'] = np.log1p(area)
    row['lat_round'] = str(int(round(geo_lat, 1)))
    row['lon_round'] = str(int(round(geo_lon, 1)))

    data = pd.DataFrame([row])
    for feature in CAT_FEATURES:
        if feature in data.columns:
            data[feature] = data[feature].fillna('-1').astype(str)

    return data[FEATURES]


def predict_price(flat):
    data = make_features(flat)

    price = np.expm1(model.predict(data)[0])
    low = np.expm1(model_low.predict(data)[0])
    high = np.expm1(model_high.predict(data)[0])
    low, high = min(low, high), max(low, high)

    if not low <= price <= high:
        width = max(high - low, price * 0.25)
        low = price - width / 2
        high = price + width / 2

    return round(price), round(max(low, 0)), round(max(high, 0))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        'Привет! Я оцениваю квартиру по данным о квартирах в России.\n\n'
        'Выбери регион или напиши номер региона, например 77 для Москвы.',
        reply_markup=ReplyKeyboardMarkup(
            [
                ['Москва', 'Санкт-Петербург'],
                ['Московская область', 'Краснодарский край'],
                ['Свердловская область', 'Татарстан'],
            ],
            one_time_keyboard=True,
            resize_keyboard=True,
        ),
    )
    return REGION


async def get_region(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    region = REGION_CODES.get(text, text)

    if not region.isdigit():
        await update.message.reply_text('Напиши регион текстом из кнопок или числом, например 77')
        return REGION

    context.user_data['id_region'] = region
    await update.message.reply_text(
        'Какая общая площадь всей квартиры? Например: 60 кв. м',
        reply_markup=ReplyKeyboardRemove(),
    )
    return AREA

async def get_area(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        area = float(update.message.text.replace(',', '.'))

        context.user_data['area'] = area
        await update.message.reply_text('Сколько комнат? Студия = 0, обычная двушка = 2')
        return ROOMS
    except ValueError:
        await update.message.reply_text('Введи число, например: 60')
        return AREA


async def get_rooms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        rooms = int(update.message.text)
        if rooms < 0 or rooms > 6:
            await update.message.reply_text('Введи число комнат от 0 до 6')
            return ROOMS

        context.user_data['rooms'] = rooms
        await update.message.reply_text('На каком этаже квартира? Например: 7')
        return FLOOR
    except ValueError:
        await update.message.reply_text('Введи целое число')
        return ROOMS


async def get_floor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        level = int(update.message.text)
        if level < 1 or level > 80:
            await update.message.reply_text('Введи этаж от 1 до 80')
            return FLOOR

        context.user_data['level'] = level
        await update.message.reply_text('Сколько всего этажей в доме? Например: 16')
        return TOTAL_FLOORS
    except ValueError:
        await update.message.reply_text('Введи целое число')
        return FLOOR


async def get_total_floors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        levels = int(update.message.text)
        if levels < context.user_data['level']:
            await update.message.reply_text('Всего этажей не может быть меньше этажа квартиры')
            return TOTAL_FLOORS
        if levels > 80:
            await update.message.reply_text('Введи этажность до 80')
            return TOTAL_FLOORS

        context.user_data['levels'] = levels
        await update.message.reply_text('Какая площадь кухни отдельно? Если не знаешь, напиши 0')
        return KITCHEN
    except ValueError:
        await update.message.reply_text('Введи целое число')
        return TOTAL_FLOORS


async def get_kitchen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        kitchen_area = float(update.message.text.replace(',', '.'))
        area = context.user_data['area']

        if kitchen_area <= 0:
            kitchen_area = max(5, min(area * 0.16, 20))
        if kitchen_area > area * 0.7:
            await update.message.reply_text('Кухня не может быть больше почти всей квартиры. Введи еще раз')
            return KITCHEN

        context.user_data['kitchen_area'] = kitchen_area
        await update.message.reply_text(
            'Тип дома? Напиши цифру:\n'
            '0 - не знаю\n1 - панельный\n2 - монолитный\n3 - кирпичный\n4 - блочный\n5 - деревянный\n6 - другой'
        )
        return BUILDING_TYPE
    except ValueError:
        await update.message.reply_text('Введи число, например: 10')
        return KITCHEN


async def get_building_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    building_type = update.message.text.strip()
    if building_type not in BUILDING_TYPES:
        await update.message.reply_text('Введи цифру от 0 до 6')
        return BUILDING_TYPE

    context.user_data['building_type'] = building_type
    await update.message.reply_text('Тип квартиры? 0 - вторичка, 2 - новостройка')
    return OBJECT_TYPE


async def get_object_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    object_type = update.message.text.strip()
    if object_type not in OBJECT_TYPES:
        await update.message.reply_text('Введи 0 для вторички или 2 для новостройки')
        return OBJECT_TYPE

    context.user_data['object_type'] = object_type
    await update.message.reply_text('Почтовый индекс? Если не знаешь, напиши 0')
    return POSTAL_CODE


async def get_postal_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    postal_code = update.message.text.strip()
    if postal_code == '0':
        postal_code = ''
    elif not postal_code.isdigit():
        await update.message.reply_text('Индекс должен быть числом. Если не знаешь, напиши 0')
        return POSTAL_CODE

    context.user_data['postal_code'] = postal_code
    await update.message.reply_text(
        'Координаты квартиры: широта и долгота через пробел.\n'
        'Например: 55.75 37.62\n'
        'Если не знаешь, напиши 0, я возьму средние по региону.'
    )
    return COORDS


async def get_coords(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text.strip().replace(',', '.')
        if text != '0':
            parts = text.split()
            if len(parts) != 2:
                await update.message.reply_text('Нужно две цифры через пробел или просто 0')
                return COORDS

            geo_lat = float(parts[0])
            geo_lon = float(parts[1])
            if not 40 <= geo_lat <= 75 or not 15 <= geo_lon <= 185:
                await update.message.reply_text('Координаты выглядят странно. Введи еще раз или напиши 0')
                return COORDS

            context.user_data['geo_lat'] = geo_lat
            context.user_data['geo_lon'] = geo_lon

        flat = dict(context.user_data)
        price, low, high = predict_price(flat)

        await update.message.reply_text(
            'Оценка квартиры по input_data\n\n'
            f'Регион: {flat["id_region"]}\n'
            f'Площадь: {flat["area"]} кв. м\n'
            f'Комнат: {flat["rooms"]}\n'
            f'Этаж: {flat["level"]} из {flat["levels"]}\n'
            f'Кухня: {flat["kitchen_area"]:.1f} кв. м\n'
            f'Тип дома: {BUILDING_TYPES[flat["building_type"]]}\n'
            f'Тип объекта: {OBJECT_TYPES[flat["object_type"]]}\n\n'
            f'Примерная цена: {price:,.0f} руб.\n'
            f'Интервал: {low:,.0f} - {high:,.0f} руб.',
            reply_markup=ReplyKeyboardRemove(),
        )
        await update.message.reply_text('Чтобы оценить ещё одну квартиру, напиши /start')
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text('Введи координаты как числа или напиши 0')
        return COORDS
    except Exception as error:
        logger.exception('Ошибка при расчете цены')
        await update.message.reply_text(f'Ошибка: {error}')
        return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Отменено. Чтобы начать заново, напиши /start',
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def unknown_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Напиши /start, чтобы начать оценку квартиры.')


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception('Ошибка при обработке update=%s', update, exc_info=context.error)

    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            'Что-то пошло не так. Напиши /start, чтобы начать заново.'
        )


def main():
    if not BOT_TOKEN:
        raise ValueError('Не найден BOT_TOKEN в .env')

    lock_single_instance()

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .concurrent_updates(False)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )
    conv = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            REGION: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_region)],
            AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_area)],
            ROOMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_rooms)],
            FLOOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_floor)],
            TOTAL_FLOORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_total_floors)],
            KITCHEN: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_kitchen)],
            BUILDING_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_building_type)],
            OBJECT_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_object_type)],
            POSTAL_CODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_postal_code)],
            COORDS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_coords)],
        },
        fallbacks=[CommandHandler('start', start), CommandHandler('cancel', cancel)],
        allow_reentry=True,
        per_chat=True,
        per_user=True,
    )
    app.add_handler(conv)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_message))
    app.add_error_handler(error_handler)
    logger.info('Бот запущен. Модель: input_data.csv')
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        bootstrap_retries=-1,
        drop_pending_updates=True,
    )


if __name__ == '__main__':
    main()
