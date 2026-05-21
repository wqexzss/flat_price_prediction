import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


OUTPUT_PATH = Path('data/raw/flats_from_sites.csv')
BASE_URL = 'https://realty.yandex.ru'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

CITY_URLS = {
    'Москва': 'moskva',
    'Санкт-Петербург': 'sankt-peterburg',
}

METRO_DISTANCE = {
    'Арбатская': 1,
    'Кропоткинская': 1,
    'Смоленская': 2,
    'Цветной бульвар': 2,
    'Павелецкая': 3,
    'Крестьянская застава': 4,
    'Фрунзенская': 4,
    'Тульская': 5,
    'Выставочная': 5,
    'Шелепиха': 6,
    'Савёловская': 6,
    'Дмитровская': 7,
    'Аэропорт': 7,
    'Дубровка': 7,
    'Крымская': 7,
    'ЗИЛ': 7,
    'Нагатинская': 8,
    'Технопарк': 8,
    'Кожуховская': 8,
    'Алексеевская': 8,
    'Хорошёво': 9,
    'Бутырская': 9,
    'Минская': 9,
    'Народное Ополчение': 9,
    'ВДНХ': 10,
    'Фонвизинская': 10,
    'Давыдково': 11,
    'Терехово': 11,
    'Водный стадион': 11,
    'Аминьевская': 12,
    'Матвеевская': 12,
    'Крылатское': 13,
    'Братиславская': 14,
    'Отрадное': 14,
    'Юго-Западная': 14,
    'Москворечье': 16,
    'Беломорская': 17,
    'Новодачная': 18,
    'Саларьево': 21,
    'Филатов Луг': 24,
    'Ольгино': 25,
}


def get_int(text):
    digits = re.sub(r'\D', '', text)
    return int(digits) if digits else None


def get_float(text):
    return float(text.replace(',', '.'))


def find_price(text):
    prices = []

    for value in re.findall(r'(\d[\d\s]{3,})\s*₽', text):
        price = get_int(value)

        if price and 1000000 <= price <= 500000000:
            prices.append(price)

    if not prices:
        return None

    return min(prices)


def find_metro(text):
    match = re.search(r'([А-ЯЁа-яёA-Za-z\-\s]+)\s+\d+\s+мин\.', text)

    if not match:
        return None

    metro = match.group(1).strip()
    metro = re.sub(r'.*?(\b[А-ЯЁ][А-ЯЁа-яё\-\s]+)$', r'\1', metro)
    metro = metro.strip()

    if metro in ['года', 'квартал']:
        return None

    return metro


def get_distance_from_center(metro):
    if metro in METRO_DISTANCE:
        return METRO_DISTANCE[metro]
    return 10


def parse_flat(card, city):
    text = card.get_text(' ', strip=True)
    link_tag = card.find('a', href=True)

    area = re.search(r'(\d+(?:[,.]\d+)?)\s*м²', text)
    rooms = re.search(r'(\d+)-комнат', text)
    floor = re.search(r'(\d+)\s*этаж\s*из\s*(\d+)', text)
    metro = find_metro(text)

    link = None
    if link_tag:
        link = urljoin(BASE_URL, link_tag['href'])

    price = find_price(text)
    area_value = get_float(area.group(1)) if area else None
    rooms_value = int(rooms.group(1)) if rooms else None
    floor_value = int(floor.group(1)) if floor else None
    total_floors = int(floor.group(2)) if floor else None

    flat = {
        'source': 'yandex',
        'city': city,
        'price': price,
        'area': area_value,
        'rooms': rooms_value,
        'floor': floor_value,
        'total_floors': total_floors,
        'metro': metro,
        'distance_from_center': get_distance_from_center(metro),
        'link': link,
        'raw_text': text,
    }

    return add_features(flat)


def add_features(flat):
    price = flat.get('price')
    area = flat.get('area')
    floor = flat.get('floor')
    total_floors = flat.get('total_floors')

    flat['price_per_meter'] = round(price / area, 2) if price and area else None
    flat['is_first_floor'] = int(floor == 1) if floor else None
    flat['is_last_floor'] = int(floor == total_floors) if floor and total_floors else None
    flat['floor_ratio'] = round(floor / total_floors, 3) if floor and total_floors else None

    return flat


def clean_data(df):
    if df.empty:
        return df

    if 'price_per_meter' not in df:
        df['price_per_meter'] = None
    if 'is_first_floor' not in df:
        df['is_first_floor'] = None
    if 'is_last_floor' not in df:
        df['is_last_floor'] = None
    if 'floor_ratio' not in df:
        df['floor_ratio'] = None

    df['price_per_meter'] = df['price_per_meter'].fillna(
        (df['price'] / df['area']).round(2)
    )
    df['is_first_floor'] = df['is_first_floor'].fillna((df['floor'] == 1).astype(int))
    df['is_last_floor'] = df['is_last_floor'].fillna(
        (df['floor'] == df['total_floors']).astype(int)
    )
    df['floor_ratio'] = df['floor_ratio'].fillna(
        (df['floor'] / df['total_floors']).round(3)
    )

    df = df.drop_duplicates(subset=['link'])
    df = df.dropna(subset=['price', 'area', 'rooms', 'floor', 'total_floors'])
    df = df[df['price'] > 1000000]
    df = df[df['area'].between(10, 350)]
    df = df[df['rooms'].between(1, 8)]
    df = df[df['floor'] > 0]
    df = df[df['total_floors'] >= df['floor']]
    df = df[df['price_per_meter'].between(150000, 2500000)]

    return df.reset_index(drop=True)


def load_old_data(output_path):
    output_path = Path(output_path)

    if not output_path.exists():
        return pd.DataFrame()

    return pd.read_csv(output_path, quotechar=chr(39))


def save_data(flats, output_path, append=True):
    new_df = pd.DataFrame(flats)
    old_df = load_old_data(output_path) if append else pd.DataFrame()
    df = pd.concat([old_df, new_df], ignore_index=True)
    df = clean_data(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quotechar=chr(39))

    print('Новых объявлений:', len(new_df))
    print('Всего после очистки:', len(df))
    print('Файл:', output_path)
    return df


def parse_yandex(cities, pages=5, output_path=OUTPUT_PATH, append=True):
    flats = []

    for city in cities:
        city_url = CITY_URLS[city]

        for page in range(1, pages + 1):
            print(f'{city}, страница {page}')

            url = f'{BASE_URL}/{city_url}/kupit/kvartira/?page={page}'
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'lxml')
            cards = soup.select('li.OffersSerpItem')

            print('Найдено карточек:', len(cards))

            for card in cards:
                flat = parse_flat(card, city)

                if flat['price'] and flat['area'] and flat['rooms']:
                    flats.append(flat)

            time.sleep(2)

    return save_data(flats, output_path, append=append)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', nargs='+', default=['Москва'], choices=list(CITY_URLS))
    parser.add_argument('--pages', type=int, default=5)
    parser.add_argument('--output', default=str(OUTPUT_PATH))
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()

    parse_yandex(
        cities=args.city,
        pages=args.pages,
        output_path=args.output,
        append=not args.replace,
    )
