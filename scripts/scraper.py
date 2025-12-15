import requests
from bs4 import BeautifulSoup
import time
import re
import xml.etree.ElementTree as ET
import json
import pandas as pd
import isodate 
import html
import uuid
from fractions import Fraction
import math

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
    
def float_to_cooking_fraction(val):
    """
    Converts float values to cooking fractions.
    e.g, 0.33333 -> 1/3, 1.5 -> 1 1/2, 0.5 -> 1/2
    """
    if val == 0: return "0"
    
    # If very close to an integer, round it
    if abs(val - round(val)) < 0.01:
        return str(round(val))
    
    # Separate integer and decimal parts
    int_part = int(val)
    dec_part = val - int_part
    
    # Limit denominator to standard cooking measures (max 12)
    frac = Fraction(dec_part).limit_denominator(12) 
    
    if frac.denominator == 1: 
        return str(int_part + frac.numerator)
        
    numerator = frac.numerator
    denominator = frac.denominator
    
    if int_part > 0:
        return f"{int_part} {numerator}/{denominator}"
    else:
        return f"{numerator}/{denominator}"

def format_ingredient_string(text):
    """
    Identifies the number at the start of an ingredient string and converts it to a fraction.
    e.g, "0.5 cup sugar" -> "1/2 cup sugar"
    """
    if not isinstance(text, str): return text
    
    # Regex: Capture the leading float or integer
    match = re.match(r"^(\d+(\.\d+)?)", text)
    if match:
        num_str = match.group(1)
        try:
            val = float(num_str)
            formatted_qty = float_to_cooking_fraction(val)
            # Replace only the first occurrence
            return text.replace(num_str, formatted_qty, 1)
        except:
            return text
    return text

def fetch_recipe_data(url):
    soup = get_soup(url)
    if not soup: return None
    result = {'soup': soup, 'json': None}
    
    script = soup.find('script', {'type': 'application/ld+json'})
    if script:
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                for item in data:
                    if 'Recipe' in item.get('@type', []):
                        result['json'] = item
                        break
            elif isinstance(data, dict):
                 if 'Recipe' in data.get('@type', []): result['json'] = data
        except: pass
    return result

def process_recipes_to_final_format(raw_results_list, FDA_DAILY_VALUES, TAG_REPLACEMENTS):
    processed_rows = []
    
    for raw_item in raw_results_list:
        json_data = raw_item.get('json')
        soup = raw_item.get('soup')
        if not json_data: continue 
        
        row = json_data.copy()
        
        # Generate UUID
        row['id'] = str(uuid.uuid4())
        
        # Parse Time
        def parse_minutes(iso_str):
            if not isinstance(iso_str, str): return 0
            try:
                val = int(isodate.parse_duration(iso_str).total_seconds() / 60)
                return val if val < 1440 else 0
            except: return 0
        row['minutes'] = parse_minutes(row.get('totalTime'))

        # Parse Nutrition
        nutrition_map = {
            'calories': ['calories'], 'total_fat_g': ['fatContent'], 'sugar_g': ['sugarContent'],
            'sodium_mg': ['sodiumContent'], 'protein_g': ['proteinContent'], 
            'sat_fat_g': ['saturatedFatContent'], 'carbs_g': ['carbohydrateContent']
        }
        nutri_data = row.get('nutrition', {})
        if isinstance(nutri_data, str):
            import ast
            try: nutri_data = ast.literal_eval(nutri_data)
            except: nutri_data = {}
            
        for target, keys in nutrition_map.items():
            row[target] = 0.0
            for k in keys:
                val = nutri_data.get(k)
                match = re.search(r"(\d+(\.\d+)?)", str(val))
                if match:
                    row[target] = float(match.group(1))
                    break
        
        # Calculate PDV
        for pdv_col, (mass_col, ref_val) in FDA_DAILY_VALUES.items():
            row[pdv_col] = round((row[mass_col] / ref_val) * 100, 1)

        # Generate Tags
        tags = set()
        def clean_and_add_tag(val):
            if pd.isna(val) or val is None or val == "": return
            s_val = str(val).lower().strip()
            if s_val in ["nan", "recipe", "recipes"]: return
            s_val = s_val.replace(" recipes", "").replace(" recipe", "")
            final_tag = TAG_REPLACEMENTS.get(s_val, s_val)
            tags.add(final_tag)

        cats = row.get('recipeCategory')
        if isinstance(cats, list): 
            for c in cats: clean_and_add_tag(c)
        else: clean_and_add_tag(cats)
        
        cuis = row.get('recipeCuisine')
        if isinstance(cuis, list):
            for c in cuis: clean_and_add_tag(c)
        else: clean_and_add_tag(cuis)
        
        if soup:
            breadcrumbs = soup.select('.mntl-breadcrumbs__link')
            for b in breadcrumbs: clean_and_add_tag(b.get_text())

        row['tags'] = list(tags)

        # Cleanup & Format Ingredients
        row['name'] = row.get('name', 'Unknown Recipe')
        if pd.isna(row['name']): row['name'] = 'Unknown'
        
        raw_ingreds = row.get('recipeIngredient', [])
        clean_ingreds = []
        for ing in raw_ingreds:
            clean_ingreds.append(format_ingredient_string(ing))
            
        row['ingredients'] = clean_ingreds
        row['n_ingredients'] = len(clean_ingreds)
        
        # Format Steps
        raw_steps = row.get('recipeInstructions', [])
        clean_steps = []
        if isinstance(raw_steps, list):
            for s in raw_steps:
                if isinstance(s, dict) and 'text' in s: clean_steps.append(s['text'])
                elif isinstance(s, str): clean_steps.append(s)
        row['steps'] = clean_steps
        row['n_steps'] = len(clean_steps)
        
        # Description
        desc = row.get('description', '')
        row['description'] = html.unescape(str(desc)) if desc else ''

        processed_rows.append(row)

    df = pd.DataFrame(processed_rows)
    target_cols = [
        'name', 'id', 'minutes', 'tags', 'n_steps', 'steps', 'description', 
        'ingredients', 'n_ingredients', 'calories', 
        'total_fat_pdv', 'sugar_pdv', 'sodium_pdv', 'protein_pdv', 'sat_fat_pdv', 'carbs_pdv', 
        'total_fat_g', 'sugar_g', 'sodium_mg', 'protein_g', 'sat_fat_g', 'carbs_g'
    ]
    
    if not df.empty:
        for col in target_cols:
            if col not in df.columns: df[col] = None
        return df[target_cols]
    else:
        return pd.DataFrame(columns=target_cols)