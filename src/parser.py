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
    'Тульская': 5,
    'Павелецкая': 3,
    'Крестьянская застава': 4,
    'Фрунзенская': 4,
    'Шелепиха': 6,
    'Выставочная': 5,
    'Белорусская': 4,
    'Савёловская': 6,
    'Дмитровская': 7,
    'Аэропорт': 7,
    'Нагатинская': 8,
    'Технопарк': 8,
    'Кожуховская': 8,
    'Дубровка': 7,
    'Крымская': 7,
    'ЗИЛ': 7,
    'Хорошёво': 9,
    'Алексеевская': 8,
    'Бутырская': 9,
    'ВДНХ': 10,
    'Минская': 9,
    'Давыдково': 11,
    'Народное Ополчение': 9,
    'Терехово': 11,
    'Водный стадион': 11,
    'Крылатское': 13,
    'Братиславская': 14,
    'Аминьевская': 12,
    'Отрадное': 14,
    'Фонвизинская': 10,
    'Юго-Западная': 14,
    'Беломорская': 17,
    'Саларьево': 21,
    'Филатов Луг': 24,
    'Москворечье': 16,
    'Матвеевская': 12,
    'Новодачная': 18,
    'Ольгино': 25,
}


def get_int(text):
    digits = re.sub(r'\D', '', text)
    return int(digits) if digits else None


def get_float(text):
    return float(text.replace(',', '.'))


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

    price = re.search(r'(\d[\d\s]{3,})\s*₽', text)
    area = re.search(r'(\d+(?:[,.]\d+)?)\s*м²', text)
    rooms = re.search(r'(\d+)-комнат', text)
    floor = re.search(r'(\d+)\s*этаж\s*из\s*(\d+)', text)
    metro = find_metro(text)

    link = None
    if link_tag:
        link = urljoin(BASE_URL, link_tag['href'])

    return {
        'source': 'yandex',
        'city': city,
        'price': get_int(price.group(1)) if price else None,
        'area': get_float(area.group(1)) if area else None,
        'rooms': int(rooms.group(1)) if rooms else None,
        'floor': int(floor.group(1)) if floor else None,
        'total_floors': int(floor.group(2)) if floor else None,
        'metro': metro,
        'distance_from_center': get_distance_from_center(metro),
        'link': link,
        'raw_text': text,
    }


def parse_yandex(city='Москва', pages=5, output_path=OUTPUT_PATH):
    flats = []
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

    df = pd.DataFrame(flats)

    if not df.empty:
        df = df.drop_duplicates(subset=['link'])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quotechar=chr(39))

    print('Сохранено объявлений:', len(df))
    print('Файл:', output_path)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='Москва', choices=list(CITY_URLS))
    parser.add_argument('--pages', type=int, default=5)
    parser.add_argument('--output', default=str(OUTPUT_PATH))
    args = parser.parse_args()

    parse_yandex(args.city, args.pages, args.output)
