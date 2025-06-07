# scraper.py

import os
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_page(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text


def extract_midi_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith((".mid", ".midi")):
            full_url = urljoin(base_url, href)
            links.append(full_url)
    return list(set(links))  # unique


def download_file(url, save_folder="midi_data"):
    os.makedirs(save_folder, exist_ok=True)
    filename = os.path.basename(urlparse(url).path)
    path = os.path.join(save_folder, filename)
    if os.path.exists(path):
        return  # bereits vorhanden
    session = requests.Session()
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    finally:
        session.close()


def parallel_download(urls, max_workers=os.cpu_count() * 5):
    """
    Lädt URLs parallel mit einem ThreadPool basierend auf der CPU-Kernezahl.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url): url for url in urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
            url = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Failed {url}: {e}")


def scrape_midi_from_pages(page_list_file): # TODO: save_folder_name einbauen!
    midi_urls = []
    with open(page_list_file, "r") as f:
        pages = [line.strip() for line in f if line.strip().startswith("http")]
    for page in pages:
        try:
            html = fetch_page(page)
            midi_urls += extract_midi_links(html, page)
        except Exception as e:
            print(f"Error fetching {page}: {e}")
    midi_urls = list(set(midi_urls))
    print(f"Found {len(midi_urls)} MIDI files.")
    # Paralleler Download basierend auf CPU-Kernen
    parallel_download(midi_urls)


if __name__ == "__main__":
    scrape_midi_from_pages(r"vgmusic_midi_scraper-master\vgmusic_midi_scraper-master\vgmusic_sources.txt") # TODO: save_folder_name="midi_data" einbauen!
