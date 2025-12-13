import json
import pandas as pd

def get_constraints(row, meal_keywords, diet_keywords, method_keywords, style_keywords):
    """
    Extracts dietary constraints from a recipe row based on keywords.
    Args:
        row (pd.Series): A row from the DataFrame representing a recipe.
        meal_keywords (list): List of meal-related keywords to look for.
        diet_keywords (list): List of diet-related keywords to look for.
    Returns:
        str: A string with constraints.
    """
    constraints = []
    tags_list = row['tags']

    # Check for meal type keywords
    for tag in tags_list:
        if tag in meal_keywords:
            constraints.append(meal_keywords[tag])

    # Check for diet type keywords
    for tag in tags_list:
        if tag in diet_keywords:
            constraints.append(diet_keywords[tag])

    # Check for cooking methods
    for tag in tags_list:
        if tag in method_keywords:
            constraints.append(method_keywords[tag])

    # Check for style keywords
    for tag in tags_list:
        if tag in style_keywords:
            constraints.append(style_keywords[tag])       

    # Time constraints
    if row['minutes'] <= 15:
        constraints.append("Very Quick")
    elif row['minutes'] <= 30:
        constraints.append("Quick")
    elif row['minutes'] >= 90:
        constraints.append("Slow Cooked")

    # Difficulty constraints
    if row['n_steps'] <= 5:
        constraints.append("Easy")
    elif row['n_steps'] >= 15:
        constraints.append("Complex")

    if not constraints:
        return "General Dish"

    return ", ".join(constraints)


def format_recipe_body(row):
    """
    Formats the target output.
    """

    ingredients_string = "\n- ".join(row['ingredients'])

    steps_string = ""
    for i, step in enumerate(row['steps'], start=1):
        steps_string += f"{i}. {step.capitalize()}\n"

    return f"**{row['name'].title()}**\n\nIngredients:\n- {ingredients_string}\n\n**Instructions:**\n{steps_string}"
    

def generate_llm_dataset(df, meal_map, diet_map, method_map, style_map, max_length=2048):
    """
    Generates a dataset suitable for LLM training.
    Args:
        df (pd.DataFrame): The cleaned recipe DataFrame.
        meal_map (dict): Mapping of meal keywords to constraints.
        diet_map (dict): Mapping of diet keywords to constraints.
        max_length (int): Maximum length of the formatted recipe body.
    Returns:
        formatted_data (list): List of dictionaries ready for JSONL format.
    """

    formatted_data = []
    skipped_count = 0

    for _, row in df.iterrows():

        constraints = get_constraints(row, meal_map, diet_map, method_map, style_map)
        output_text = format_recipe_body(row)

        # Check length constraint
        if len(output_text) > max_length:
            skipped_count += 1
            continue

        entry = {
            "instruction": "You are a smart chef. Generate a recipe that uses the provided ingredients and strictly follows the context constraints.",
            "input": f"Ingredients: {', '.join(row['ingredients'])}. Context: {constraints}",
            "output": output_text + " </s>"
        }

        formatted_data.append(entry)

    print(f"   - Processed {len(formatted_data)} items.")
    print(f"   - Skipped {skipped_count} items due to length > {max_length}.")

    return formatted_data