import chromadb
from chromadb.utils import embedding_functions
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

class ChefTools:
    def __init__(self):
        # Paths
        self.base_path = os.path.dirname(__file__)
        self.db_path = os.path.join(self.base_path, "../chroma_db")
        self.json_path = os.path.join(self.base_path, "../data/knowledge/culinary_constants.json")

        # Embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

        # ChromaDB connection
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Getting collection
        self.recipe_collection = self.client.get_collection("chef_cookbook")
        self.safety_collection = self.client.get_collection("chef_safety")

        # Loading constants JSON to memory
        try:
            with open(self.json_path, 'r') as file:
                self.constants_data = json.load(file)
        except FileNotFoundError:
            print(f"  Constants file not found at {self.json_path}")
            self.constants_data = {}

    def get_recipes(self, query, n_results=3):
        """
        Searches the recipe collection.
        Returns a formatted string of top N recipes.
        """

        # Embedding the query, then turning into list for ChromaDB
        query_embedding = self.embedder.encode([query]).tolist()

        # Querying the collection
        results = self.recipe_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas"]
        )

        # Parsing results
        metadatas = results['metadatas'][0]

        if not metadatas:
            return ""
        
        # Formatting to string for Brain model
        formatted_results = ""
        for i, meta in enumerate(metadatas):
            name = meta.get('name', 'Unknown Dish')
            ingredients = meta.get('ingredients', 'N/A')
            steps = meta.get('steps', 'N/A')

            formatted_results += f"""
--- RECIPE OPTION {i+1}: {name} ---
Ingredients: {ingredients}
Instructions: {steps}
-----------------------------------
"""

        return formatted_results.strip()
    
    def check_safety(self, query, n_results=2):
        """
        Searches the safety collection.
        Returns a string of relevant safety rules.
        """

        # Embedding the query, turning it into list for ChromaDB
        query_embedding = self.embedder.encode([query]).tolist()

        # Querying the collection
        results = self.safety_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents"]
        )

        # Formatting output using documents
        documents = results['documents'][0]

        if not documents:
            return ""
        
        return "\n".join([f"RULE: {doc}" for doc in documents])
    
    def search_constants(self, query):
        """
        Searches the JSON data for conversions and substitutions.
        """

        query_lower = query.lower()
        found_items = []

        # String matching
        for key, value in self.constants_data.items():
            if query_lower in key.lower():
                found_items.append(f"{key}: {value}")

        return "\n".join(found_items[:3]) if found_items else ""

