import requests
from bs4 import BeautifulSoup
import time
import re

HEADER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

visited_urls = set()
final_category_urls = set()
final_recipe_urls = set()

def get_clean_slug(url):
    """
    Removes domain and recipe IDs to get the pure category path.
    Example: '.../recipes/113/appetizers-and-snacks/pastries/' 
    Returns: 'appetizers-and-snacks/pastries/'
    """
    # This regex looks for '/recipes/' followed by digits, and captures everything after
    match = re.search(r'/recipes/\d+/(.+)', url)
    if match:
        return match.group(1)
    return ""

def get_soup(url):
    """Takes an URL and returns a soup object."""
    try:
        #sending request to the website
        response = requests.get(url, headers=HEADER, timeout=10)

        #check status
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
        else:
            print(f"Failed to retrieve {url}: Status code {response.status_code}")

    except Exception as e:
        print(f"An error occurred while fetching: {e}")

    return None

def category_crawler(url):
    """Crawls category pages to save recipe sub-categories."""
    if url in visited_urls:
        return
    
    visited_urls.add(url)

    soup = get_soup(url)
    if not soup:
        return
    
    main = soup.find('main')
    if not main:
        main = soup  # Fallback to entire soup if <main> not found

    current_slug = get_clean_slug(url)

    has_subcategories = False

    all_links = main.find_all('a', href=True)

    for link in all_links:
        href = link['href']

        child_slug = get_clean_slug(href)

        if (current_slug and child_slug 
            and child_slug.startswith(current_slug) 
            and child_slug != current_slug):

            has_subcategories = True
            category_crawler(href)
            time.sleep(0.5)

    if not has_subcategories:
        print(f"Leaf category found: {url}")
        final_category_urls.add(url)