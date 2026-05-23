import argparse
import random
import re
import time
from pathlib import Path
from urllib.parse import urlencode, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


RAW_FILE = Path('data/raw/flats_from_sites.csv')
CLEAN_FILE = Path('data/processed/flats_clean.csv')

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept-Language': 'ru-RU,ru;q=0.9',
}

SITES = {
    'yandex': 'https://realty.yandex.ru',
    'domclick': 'https://domclick.ru',
}

URLS = {
    'Москва': {
        'yandex': 'https://realty.yandex.ru/moskva/kupit/kvartira/',
        'domclick': 'https://domclick.ru/pokupka/kvartiry',
    },
    'Санкт-Петербург': {
        'yandex': 'https://realty.yandex.ru/sankt-peterburg/kupit/kvartira/',
        'domclick': 'https://domclick.ru/pokupka/kvartiry/sankt-peterburg',
    },
}

METRO_DIST = {
    'Арбатская': 1,
    'Кропоткинская': 1,
    'Смоленская': 2,
    'Павелецкая': 3,
    'Тульская': 5,
    'Выставочная': 5,
    'Шелепиха': 6,
    'Аэропорт': 7,
    'Дмитровская': 7,
    'Алексеевская': 8,
    'Нагатинская': 8,
    'ВДНХ': 10,
    'Белорусская': 10,
    'Новаторская': 10,
    'Давыдково': 11,
    'Аминьевская': 12,
    'Крылатское': 13,
    'Отрадное': 14,
    'Юго-Западная': 14,
    'Москворечье': 16,
    'Беломорская': 17,
    'Саларьево': 21,
}


def fix_text(text):
    text = str(text or '').replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def only_int(text):
    text = re.sub(r'\D', '', str(text))
    return int(text) if text else None


def to_float(text):
    try:
        return float(str(text).replace(',', '.').replace(' ', ''))
    except ValueError:
        return None


def has_words(text, words):
    text = str(text).lower()
    return int(any(word in text for word in words))


def get_price(text):
    text = fix_text(text)
    prices = []

    for value in re.findall(r'(\d[\d\s]{4,})\s*(?:₽|руб|р\.)', text, flags=re.I):
        price = only_int(value)
        if price and 1_000_000 <= price <= 500_000_000:
            prices.append(price)

    return min(prices) if prices else None


def get_area(text):
    match = re.search(r'(\d+(?:[,.]\d+)?)\s*(?:м²|м2|кв\.?\s*м)', text, flags=re.I)
    return to_float(match.group(1)) if match else None


def get_rooms(text):
    text = str(text).lower()
    if 'студия' in text:
        return 1

    match = re.search(r'(\d+)\s*[- ]?\s*(?:комнат|комн|к\.|к\b)', text)
    return int(match.group(1)) if match else None


def get_floor(text):
    match = re.search(r'(\d+)\s*(?:этаж|эт\.?)\s*(?:из|/)\s*(\d+)', text, flags=re.I)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(r'(\d+)\s*/\s*(\d+)\s*эт', text, flags=re.I)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


def get_metro_minutes(text):
    for value in re.findall(r'(\d{1,2})\s*мин\.?', fix_text(text)):
        value = int(value)
        if 1 <= value <= 60:
            return value
    return None


def get_metro(text):
    match = re.search(r'([А-ЯЁа-яёA-Za-z\-\s]+?)\s+\d{1,2}\s*мин\.?', fix_text(text))
    if not match:
        return 'unknown'

    metro = match.group(1).strip()
    metro = re.sub(r'.*?(\b[А-ЯЁ][А-ЯЁа-яё\-\s]+)$', r'\1', metro).strip()
    if metro.lower() in ['года', 'квартал', 'этаж', 'корпус']:
        return 'unknown'
    return metro or 'unknown'


def get_jk(text):
    text = fix_text(text)
    match = re.search(r'ЖК\s*[«"]([^»"]+)[»"]', text, flags=re.I)
    if match:
        return match.group(1).strip()[:80]

    match = re.search(r'ЖК\s+([А-ЯЁA-Za-z0-9][А-ЯЁа-яёA-Za-z0-9\-\s]{2,50})', text)
    return match.group(1).strip()[:80] if match else 'unknown'


def dist_to_center(metro):
    return METRO_DIST.get(metro, 10)


def add_features(row):
    text = row.get('raw_text', '')
    area = row.get('area')
    rooms = row.get('rooms')
    floor = row.get('floor')
    total = row.get('total_floors')
    metro = row.get('metro') or 'unknown'

    row['metro'] = metro
    row['metro_minutes'] = row.get('metro_minutes') or 20
    row['distance_from_center'] = dist_to_center(metro)
    row['price_per_meter'] = round(row['price'] / area, 2) if row.get('price') and area else None
    row['has_metro'] = int(metro != 'unknown')
    row['is_first_floor'] = int(floor == 1) if floor else 0
    row['is_last_floor'] = int(floor == total) if floor and total else 0
    row['floor_ratio'] = round(floor / total, 3) if floor and total else 0.5
    row['area_per_room'] = round(area / rooms, 2) if area and rooms else None
    row['complex_name'] = get_jk(text)
    row['is_apartment'] = has_words(text, ['апартамент'])
    row['is_new_building'] = has_words(text, ['застройщик', 'квартал 202', 'жк ', 'жилой комплекс'])
    row['has_renovation'] = has_words(text, ['ремонт', 'отделк', 'мебел', 'дизайнер'])
    row['is_premium'] = has_words(text, ['премиум', 'элитн', 'бизнес-класс', 'бизнес класса'])
    row['has_discount'] = has_words(text, ['скидк', 'акци', 'хорошая цена'])
    return row


def parse_card(card, city, site):
    text = fix_text(card.get_text(' ', strip=True))
    link_tag = card.find('a', href=True)
    metro = get_metro(text)
    floor, total = get_floor(text)

    row = {
        'source': site,
        'city': city,
        'price': get_price(text),
        'area': get_area(text),
        'rooms': get_rooms(text),
        'floor': floor,
        'total_floors': total,
        'metro': metro,
        'metro_minutes': get_metro_minutes(text),
        'link': urljoin(SITES[site], link_tag['href']) if link_tag else None,
        'raw_text': text,
    }
    return add_features(row)


def page_url(city, site, page):
    return URLS[city][site] + '?' + urlencode({'page': page})


def load_page(url, site):
    try:
        response = requests.get(url, headers=HEADERS, timeout=25)
        response.encoding = 'utf-8'
    except requests.RequestException as error:
        print('Ошибка запроса:', error)
        return None

    if site == 'domclick' and '__qrator' in response.text.lower():
        print('Домклик открыл защиту Qrator, без браузера он не парсится')
        return None

    if response.status_code >= 400:
        print('Ошибка сайта:', response.status_code)
        return None

    return response.text


def find_cards(soup, site):
    variants = {
        'yandex': ['li.OffersSerpItem', '[data-test="offer-card"]', 'article'],
        'domclick': ['[data-testid*="offer"]', '[data-test*="offer"]', 'article'],
    }

    for selector in variants[site]:
        cards = soup.select(selector)
        if cards:
            return cards
    return []


def clean_data(df):
    if df.empty:
        return df

    text_cols = ['source', 'city', 'metro', 'complex_name', 'link', 'raw_text']
    num_cols = [
        'price', 'area', 'rooms', 'floor', 'total_floors',
        'distance_from_center', 'metro_minutes', 'price_per_meter',
        'has_metro', 'is_first_floor', 'is_last_floor', 'floor_ratio',
        'area_per_room', 'is_apartment', 'is_new_building',
        'has_renovation', 'is_premium', 'has_discount',
    ]

    for col in text_cols:
        if col not in df:
            df[col] = 'unknown'
        df[col] = df[col].fillna('unknown').astype(str)

    for col in num_cols:
        if col not in df:
            df[col] = None
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.drop_duplicates(subset=['link'])
    df = df.drop_duplicates(subset=['raw_text'])
    df = df.dropna(subset=['price', 'area', 'rooms', 'floor', 'total_floors'])

    df = df[df['price'].between(1_000_000, 200_000_000)]
    df = df[df['area'].between(10, 250)]
    df = df[df['rooms'].between(1, 7)]
    df = df[df['floor'] > 0]
    df = df[df['total_floors'] >= df['floor']]

    df['price_per_meter'] = (df['price'] / df['area']).round(2)
    df = df[df['price_per_meter'].between(120_000, 2_000_000)]

    df['metro_minutes'] = df['metro_minutes'].fillna(20).clip(1, 60)
    df['distance_from_center'] = df['metro'].apply(dist_to_center)
    df['has_metro'] = (df['metro'] != 'unknown').astype(int)
    df['is_first_floor'] = (df['floor'] == 1).astype(int)
    df['is_last_floor'] = (df['floor'] == df['total_floors']).astype(int)
    df['floor_ratio'] = (df['floor'] / df['total_floors']).round(3)
    df['area_per_room'] = (df['area'] / df['rooms']).round(2)
    df['complex_name'] = df['raw_text'].apply(get_jk)
    df['is_apartment'] = df['raw_text'].apply(lambda x: has_words(x, ['апартамент']))
    df['is_new_building'] = df['raw_text'].apply(lambda x: has_words(x, ['застройщик', 'квартал 202', 'жк ']))
    df['has_renovation'] = df['raw_text'].apply(lambda x: has_words(x, ['ремонт', 'отделк', 'мебел']))
    df['is_premium'] = df['raw_text'].apply(lambda x: has_words(x, ['премиум', 'элитн', 'бизнес-класс']))
    df['has_discount'] = df['raw_text'].apply(lambda x: has_words(x, ['скидк', 'акци', 'хорошая цена']))

    return df.reset_index(drop=True)


def save_result(rows, output, clean_output, append=True):
    new_df = pd.DataFrame(rows)

    if append and Path(output).exists():
        old_df = pd.read_csv(output, quotechar=chr(39))
        df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        df = new_df

    df = clean_data(df)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(clean_output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, quotechar=chr(39))
    df.to_csv(clean_output, index=False)

    print('Новых объявлений:', len(new_df))
    print('Всего после очистки:', len(df))
    print('Файл:', output)


def parse_sites(cities, sites, pages, output, clean_output, append=True, delay=1.5):
    rows = []

    for city in cities:
        for site in sites:
            empty_pages = 0

            for page in range(1, pages + 1):
                print(site, city, 'страница', page, flush=True)
                html = load_page(page_url(city, site, page), site)

                if not html:
                    empty_pages += 1
                    if empty_pages >= 3:
                        break
                    continue

                soup = BeautifulSoup(html, 'lxml')
                page_rows = []

                for card in find_cards(soup, site):
                    row = parse_card(card, city, site)
                    if row['price'] and row['area'] and row['rooms'] and row['floor']:
                        page_rows.append(row)

                print('Найдено:', len(page_rows), flush=True)
                rows.extend(page_rows)
                empty_pages = 0 if page_rows else empty_pages + 1

                if empty_pages >= 3:
                    break
                time.sleep(delay + random.random())

    save_result(rows, output, clean_output, append=append)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', nargs='+', default=['Москва'], choices=list(URLS))
    parser.add_argument('--sites', nargs='+', default=['yandex'], choices=list(SITES))
    parser.add_argument('--pages', type=int, default=50)
    parser.add_argument('--output', default=str(RAW_FILE))
    parser.add_argument('--processed-output', default=str(CLEAN_FILE))
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--delay', type=float, default=1.5)
    args = parser.parse_args()

    parse_sites(
        cities=args.city,
        sites=args.sites,
        pages=args.pages,
        output=args.output,
        clean_output=args.processed_output,
        append=not args.replace,
        delay=args.delay,
    )


if __name__ == '__main__':
    main()
