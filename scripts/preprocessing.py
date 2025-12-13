import pandas as pd
import numpy as np
import ast

# Constants
NUTRITION_COLS = ['calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv',
                   'protein_pdv', 'sat_fat_pdv', 'carbs_pdv']

def parse_nutrition(df):
    """
    Parses the stringified list in the 'nutrition' column into separate float columns.
    """
    temp = df['nutrition'].str[1:-1].str.split(', ', expand=True)

    temp.columns = NUTRITION_COLS

    for col in NUTRITION_COLS:
        temp[col] = temp[col].astype(float)

    return pd.concat([df.drop(columns=['nutrition']), temp], axis=1)

def calculate_nutrition_mass(df):
    """
    Calculates the actual mass (g) or amount (mg) of nutritional components 
    based on their %DV and standard daily values.
    
    Standard Values (based on typical 2000 kcal diet labels):
    - Fat: 65g
    - Carbs: 300g
    - Protein: 50g
    - Sat Fat: 20g
    - Sodium: 2400mg
    - Sugar: 50g (Approximate)
    """
    # We Map the 'column_name' -> 'standard_mass'
    daily_values = {
        'total_fat_pdv': 65.0,
        'sugar_pdv': 50.0,
        'sodium_pdv': 2400.0,  # This results in mg
        'protein_pdv': 50.0,
        'sat_fat_pdv': 20.0,
        'carbs_pdv': 300.0
    }

    df = df.copy()

    for col, ref_value in daily_values.items():
        if col in df.columns:
            #creating new column names (special case for sodium)
            suffix = '_mg' if 'sodium' in col else '_g'
            new_col = col.replace('_pdv', suffix)

            df[new_col] = (df[col] / 100) * ref_value

    df['estimated_weight_g'] = df['total_fat_g'] + df['carbs_g'] + df['protein_g']

    return df

def filter_outliers(df):
    """
    Applies physics-based filters and structural integrity checks to remove fake/broken recipes.
    - Removes 0-calorie or 0-minute items.
    - Removes items with impossible calorie density (missing ingredients).
    - Removes items with no name, ingredients, or steps.
    """

    initial_count = len(df)
    df = df.dropna(subset=['name', 'ingredients', 'steps'])

    df = df[df['name'] != '']

    print(f"   - Removed {initial_count - len(df)} items due to missing essential fields.")

    # Range Check
    mask_range = (
        (df['calories'] > 10) & 
        (df['calories'] < 5000) & 
        (df['minutes'] > 0) & 
        (df['minutes'] < 1440)
    )
    
    # 2. Density Check (Calories per Gram)
    # Avoid division by zero
    clean_mass = df['estimated_weight_g'].replace(0, np.nan)
    cal_density = df['calories'] / clean_mass
    
    # Density > 0.5 excludes "phantom" items (like the 9-cal cookie error)
    mask_density = cal_density > 0.5
    
    # Apply filters
    df_clean = df[mask_range & mask_density].copy()
    
    return df_clean

def parse_lists(df, columns=[]):
    """
    Parses stringified lists (e.g. "['a', 'b']") into actual Python lists.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    return df