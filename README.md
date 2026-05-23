# Flat price prediction

Проект для предсказания цены квартиры по параметрам.

Что сейчас есть:

- обучение модели на `input_data.csv`;
- подробная очистка датасета в `src/model.ipynb`;
- CatBoost для предсказания цены квартиры;
- отдельные модели CatBoost для нижней и верхней границы интервала;
- Telegram-бот, куда пользователь вводит параметры квартиры и получает примерную цену.

В обучении используются признаки из `input_data.csv`: площадь, комнаты, этаж,
этажность дома, кухня, координаты, регион, индекс, тип дома и тип объекта.
Telegram-бот берет ответы пользователя и собирает из них такие же признаки,
какие были при обучении.

## Как запустить

Установка зависимостей:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Файл с данными должен лежать в корне проекта:

```text
input_data.csv
```

Обучение модели находится в ноутбуке:

```text
src/model.ipynb
```

После запуска ноутбук сохраняет файлы для бота:

```text
models/input_catboost_model.cbm
models/input_catboost_low.cbm
models/input_catboost_high.cbm
models/input_model_config.json
```

На последнем обучении получились метрики на тестовой выборке:

```text
R2: 0.9058
MAPE: 13.89%
MAE: 780 781 руб.
RMSE: 1 756 825 руб.
```

Запуск Telegram-бота:

```bash
.venv/bin/python src/bot.py
```

Быстрая проверка, что код бота, зависимости и модели загружаются локально:

```bash
.venv/bin/python -c "import src.bot as b; print(b.predict_price({'id_region':'77','area':60,'rooms':2,'level':7,'levels':16,'kitchen_area':10,'building_type':'1','object_type':'0','postal_code':''}))"
```

Токен Telegram-бота должен лежать в локальном файле `.env`:

```text
BOT_TOKEN=твой_токен
```

Файл `.env` нельзя загружать на GitHub.
