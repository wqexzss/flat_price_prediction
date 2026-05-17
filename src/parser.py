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


def get_int(text):
    digits = re.sub(r'\D', '', text)
    return int(digits) if digits else None


def parse_flat(card):
    text = card.get_text(' ', strip=True)
    link_tag = card.find('a', href=True)

    price = re.search(r'(\d[\d\s]{3,})\s*₽', text)
    area = re.search(r'(\d+(?:[,.]\d+)?)\s*м²', text)
    rooms = re.search(r'(\d+)-комнат', text)
    floor = re.search(r'(\d+)\s*этаж\s*из\s*(\d+)', text)

    link = None
    if link_tag:
        link = urljoin(BASE_URL, link_tag['href'])

    return {
        'source': 'yandex',
        'price': get_int(price.group(1)) if price else None,
        'area': float(area.group(1).replace(',', '.')) if area else None,
        'rooms': int(rooms.group(1)) if rooms else None,
        'floor': int(floor.group(1)) if floor else None,
        'total_floors': int(floor.group(2)) if floor else None,
        'link': link,
        'raw_text': text,
    }


def parse_yandex(pages=1, output_path=OUTPUT_PATH):
    flats = []

    for page in range(1, pages + 1):
        print(f'Страница {page}')

        url = f'{BASE_URL}/moskva/kupit/kvartira/?page={page}'
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'lxml')
        cards = soup.select('li.OffersSerpItem')

        print('Найдено карточек:', len(cards))

        for card in cards:
            flat = parse_flat(card)
            if flat['price'] and flat['area']:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', type=int, default=1)
    parser.add_argument('--output', default=str(OUTPUT_PATH))
    args = parser.parse_args()

    parse_yandex(args.pages, args.output)
