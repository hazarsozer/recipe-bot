import requests
from bs4 import BeautifulSoup
import time
import re
import xml.etree.ElementTree as ET
import json

HEADER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

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

def sitemap_category_extract(sitemap_url, match_string):
    """
    Extracts category URLs from a sitemap that contain a specific string.
    
    Args:
        sitemap_index_url (str): The main sitemap URL (e.g., 'https://site.com/sitemap.xml')
        match_string (str): The keyword to filter by (e.g., '/appetizers-and-snacks/')
    """
    found_categories = set()

    print("🗺️  Fetching Sitemap Index...")
    try:
        response = requests.get(sitemap_url, headers=HEADER, timeout=10)

        root = ET.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        sub_sitemaps = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        print(f"🔍 Found {len(sub_sitemaps)} sub-sitemaps. Scanning for Appetizers and Snacks...")

        for sitemap in sub_sitemaps:
            if 'sitemap' not in sitemap: 
                continue

            try:
                res = requests.get(sitemap, headers=HEADER, timeout=10)
                sub_root = ET.fromstring(res.content)
                urls = [loc.text for loc in sub_root.findall('.//ns:loc', namespace)]

                for url in urls:
                    if match_string in url:
                        found_categories.add(url)

            except:
                continue

        print(f"✅ Found {len(found_categories)} URLs matching '{match_string}'.")
        return list(found_categories)
    
    except Exception as e:
        print(f"An error occurred while parsing sitemap: {e}")
        return []
    

def recipe_harvester(category_urls, url_pattern, exclude_substrings=[]):
    """
    Harvests URLs from category pages that match a specific Regex pattern.
    Args:
        category_urls (list): List of category page URLs to scrape.
        url_pattern (re.Pattern): Compiled regex pattern to match recipe URLs.
    """
    unique_recipes = set()
    total = len(category_urls)

    print(f"🚜 Starting Harvest on {total} categories...")

    for category_url in category_urls:
        print(f"📂 Processing category: {category_url}")
        soup = get_soup(category_url)
        if not soup:
            print(f"Skipping category due to fetch error: {category_url}")
            continue

        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']

            # Strip query parameters to avoid duplicates
            href = href.split('?')[0]

            if any(ex in href for ex in exclude_substrings):
                continue
                                
            if url_pattern.search(href):
                unique_recipes.add(href)

        time.sleep(0.3)

    print(f"✅ Harvested {len(unique_recipes)} unique recipes.")
    return list(unique_recipes)


def scrape_recipe_details(url):
    """
    Navigates to the given URL and extracts recipe details using JSON-LD structure.
    
    Args:
        url (str): The URL of the recipe.
    Returns:
        dict: A dictionary containing recipe details or None if failed.
    """
    soup = get_soup(url)
    if not soup:
        return None
    
    script = soup.find('script', {'type': 'application/ld+json'})
    
    if not script:
        print(f"⚠️ JSON-LD not found: {url}")
        return None

    try:
        json_content = json.loads(script.string)
        recipe_data = None

        if isinstance(json_content, list):
            for item in json_content:
                # Check if 'Recipe' is in the @type (can be list or string)
                type_val = item.get('@type', [])
                if 'Recipe' in type_val or type_val == 'Recipe':
                    recipe_data = item
                    break
        elif isinstance(json_content, dict):
             type_val = json_content.get('@type', [])
             if 'Recipe' in type_val or type_val == 'Recipe':
                    recipe_data = json_content
        
        if not recipe_data:
            return None

        # Extracting data
        data = {
            'name': recipe_data.get('name'),
            'url': url,
            'category': recipe_data.get('recipeCategory'),
            'cuisine': recipe_data.get('recipeCuisine'),
            'prep_time': recipe_data.get('prepTime'),
            'cook_time': recipe_data.get('cookTime'),
            'total_time': recipe_data.get('totalTime'),
            'yield': recipe_data.get('recipeYield'),
            'ingredients': recipe_data.get('recipeIngredient', []), # Returns a list
            'nutrition': recipe_data.get('nutrition', {}) # Returns a dict
        }

        # Clean Instructions
        instructions = recipe_data.get('recipeInstructions', [])
        clean_steps = []
        for step in instructions:
            if isinstance(step, dict) and 'text' in step:
                clean_steps.append(step['text'])
            elif isinstance(step, str):
                clean_steps.append(step)
        
        data['steps'] = clean_steps
        
        return data

    except Exception as e:
        print(f" Error occurred ({url}): {e}")
        return None