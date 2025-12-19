import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import sys

class RecipeRAGBuilder:
    def __init__(self, db_path="chroma_db", collection_name="chef_cookbook"):
        """
        Initialize the RAG Builder with database paths.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def load_data(self, file_path):
        """
        Load and inspect the dataset.
        Returns the dataframe for inspection in Notebooks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")
        
        print(f"📂 Loading data from {file_path}...")
        df = pd.read_pickle(file_path)
        
        # Basic cleaning
        df = df.fillna(0)
        df["description"] = df["description"].replace(0, "")
        
        print(f"✅ Data Loaded. Shape: {df.shape}")
        return df

    def initialize_db(self, reset=False):
        """
        Step 2: Connect to ChromaDB and prepare the collection.
        """
        print(f"⚙️ Connecting to ChromaDB at '{self.db_path}'...")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
                print(f"🗑️ Deleted existing collection: {self.collection_name}")
            except:
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Collection '{self.collection_name}' ready.")

    def ingest_batch(self, df, batch_size=500):
            if self.collection is None:
                raise ValueError("❌ DB not initialized. Run initialize_db() first.")

            print(f"🚀 Starting Ingestion of {len(df)} items...")
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]
                ids, documents, metadatas = [], [], []

                for _, row in batch.iterrows():
                    ids.append(str(row.get('id', i))) # Fallback to index if no ID
                    
                    # Searchable text
                    name = str(row.get('name', 'Untitled'))
                    desc = str(row.get('description', ''))
                    documents.append(f"{name}. {desc[:200]}")
                    
                    # Using .get(column, default_value) to prevent KeyErrors
                    meta = {
                        "name": name,
                        "minutes": int(row.get('minutes', 0)),
                        "calories": float(row.get('calories', 0)),
                        "protein_g": float(row.get('protein_g', 0)),
                        "fat_g": float(row.get('total_fat_g', 0)), # This was the crasher!
                        "sodium_mg": float(row.get('sodium_mg', 0)),
                        "ingredients": str(row.get('ingredients', '')),
                        "steps": str(row.get('steps', '')),
                        "tags": str(row.get('tags', ''))
                    }
                    metadatas.append(meta)

                self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
                if i % (batch_size * 10) == 0 and i > 0:
                    print(f"   Indexed {i} / {len(df)}...")

            print(f"🎉 Ingestion Complete. Total Count: {self.collection.count()}")