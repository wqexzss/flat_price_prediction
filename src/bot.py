import os
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ConversationHandler, ContextTypes
)
from catboost import CatBoostRegressor

BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
BOT_TOKEN = os.getenv("BOT_TOKEN")



model = CatBoostRegressor()
low_model = CatBoostRegressor()
high_model = CatBoostRegressor()

model.load_model(BASE_DIR / "models" / "catboost_model.cbm")
low_model.load_model(BASE_DIR / "models" / "catboost_low.cbm")
high_model.load_model(BASE_DIR / "models" / "catboost_high.cbm")

# Признаки: area, bedrooms, bathrooms, stories, mainroad, guestroom,
#            basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus

(AREA, BEDROOMS, BATHROOMS, STORIES,
 MAINROAD, GUESTROOM, BASEMENT,
 HOTWATER, AIRCON, PARKING, PREFAREA, FURNISHING) = range(12)

YES_NO = [["yes", "no"]]
FURNISHING_KB = [["furnished", "semi-furnished", "unfurnished"]]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я оценю стоимость квартиры.\n\n"
        "Введи площадь (кв. фут), например: 5000"
    )
    return AREA

async def get_area(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data["area"] = float(update.message.text.replace(",", "."))
        await update.message.reply_text("Сколько спален? (1–6)")
        return BEDROOMS
    except ValueError:
        await update.message.reply_text("❌ Введи число, например: 5000")
        return AREA

async def get_bedrooms(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data["bedrooms"] = int(update.message.text)
        await update.message.reply_text("Сколько ванных комнат? (1–4)")
        return BATHROOMS
    except ValueError:
        await update.message.reply_text("❌ Введи целое число")
        return BEDROOMS

async def get_bathrooms(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data["bathrooms"] = int(update.message.text)
        await update.message.reply_text("Сколько этажей в доме? (1–4)")
        return STORIES
    except ValueError:
        await update.message.reply_text("❌ Введи целое число")
        return BATHROOMS

async def get_stories(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data["stories"] = int(update.message.text)
        await update.message.reply_text(
            "Выход на главную дорогу?",
            reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
        )
        return MAINROAD
    except ValueError:
        await update.message.reply_text("❌ Введи целое число")
        return STORIES

async def get_mainroad(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["mainroad"] = update.message.text
    await update.message.reply_text(
        "Есть гостевая комната?",
        reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
    )
    return GUESTROOM

async def get_guestroom(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["guestroom"] = update.message.text
    await update.message.reply_text(
        "Есть подвал?",
        reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
    )
    return BASEMENT

async def get_basement(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["basement"] = update.message.text
    await update.message.reply_text(
        "Есть горячее водоснабжение?",
        reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
    )
    return HOTWATER

async def get_hotwater(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["hotwaterheating"] = update.message.text
    await update.message.reply_text(
        "Есть кондиционер?",
        reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
    )
    return AIRCON

async def get_aircon(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["airconditioning"] = update.message.text
    await update.message.reply_text(
        "Сколько парковочных мест? (0–3)",
        reply_markup=ReplyKeyboardMarkup([["0", "1", "2", "3"]], one_time_keyboard=True)
    )
    return PARKING

async def get_parking(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ctx.user_data["parking"] = int(update.message.text)
        await update.message.reply_text(
            "Престижный район?",
            reply_markup=ReplyKeyboardMarkup(YES_NO, one_time_keyboard=True)
        )
        return PREFAREA
    except ValueError:
        await update.message.reply_text("❌ Введи 0, 1, 2 или 3")
        return PARKING

async def get_prefarea(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["prefarea"] = update.message.text
    await update.message.reply_text(
        "Состояние меблировки?",
        reply_markup=ReplyKeyboardMarkup(FURNISHING_KB, one_time_keyboard=True)
    )
    return FURNISHING

async def get_furnishing(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["furnishingstatus"] = update.message.text
    d = ctx.user_data

    features = [[
        d["area"], d["bedrooms"], d["bathrooms"], d["stories"],
        d["mainroad"], d["guestroom"], d["basement"],
        d["hotwaterheating"], d["airconditioning"],
        d["parking"], d["prefarea"], d["furnishingstatus"]
    ]]

    try:
        price = model.predict(features)[0]
        low   = low_model.predict(features)[0]
        high  = high_model.predict(features)[0]

        await update.message.reply_text(
            f"🏠 <b>Оценка квартиры</b>\n\n"
            f"💰 <b>Прогноз:</b> {price:,.0f}\n"
            f"📊 <b>Интервал (10–90%):</b> {low:,.0f} — {high:,.0f}",
            parse_mode="HTML",
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}",
                                        reply_markup=ReplyKeyboardRemove())

    await update.message.reply_text("Хочешь оценить ещё? /start")
    return ConversationHandler.END

async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Отменено. /start — начать заново.",
                                    reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
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
            PREFAREA:  [MessageHandler(filters.TEXT & ~filters.COMMAND, get_prefarea)],
            FURNISHING:[MessageHandler(filters.TEXT & ~filters.COMMAND, get_furnishing)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(conv)
    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
