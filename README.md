# 👨‍🍳 ChefAI: The Constraint-Aware Cooking Agent

**ChefAI** is a smart cooking assistant that actually listens to your constraints.

Unlike standard AI that might suggest *“Cheeseburgers”* when you ask for a *“Vegan Dinner,”* ChefAI uses a **Dual-Agent Architecture** to separate creativity from logic. It runs locally on your computer, ensuring privacy and strict adherence to your inventory.

---

## 📂 Project Structure (What is everything?)

Here is a quick guide to understanding the code in this repository:

### 1. The Notebooks (Research & Pipeline)
* `notebooks/data_gathering.ipynb`: **Step 1.** Scrapes recipes from the web and downloads datasets. Uses `scraper.py` to extract structured JSON-LD data.
* `notebooks/eda.ipynb`: **Step 2.** The "Physics" filter. Uses `preprocessing.py` to calculate caloric density and remove "hallucinated" or broken recipes.
* `notebooks/llm_data_prep.ipynb`: **Step 3.** Prepares the training data. Uses `data_prep.py` to inject synthetic "Persona Prompts" so the model learns to chat, not just complete text.
* `notebooks/build_rag.ipynb`: **Step 4.** Builds the Vector Database. Uses `rag_builder.py` to ingest recipes and safety rules into ChromaDB.
* `notebooks/train_chef_mistral.ipynb`: **Step 5 (Track 1).** Fine-tunes **Mistral 7B** using QLoRA. This became our final "Chef" model.
* `notebooks/phi3_mini_fft.ipynb`: **Step 5 (Track 2).** Fine-tunes **Phi-3 Mini** (Full Fine-Tune). Used for comparison.
* `notebooks/evaluation_notebook.ipynb`: **Step 6.** The Test Bench. Runs both models against 100 "Hallucination Traps" to prove our system is safer.

### 2. The Scripts (Production Application)
* `scripts/scraper.py`: Helper functions for sitemap parsing and data extraction.
* `scripts/preprocessing.py`: Physics-based filtering logic (Mass/Density calculations).
* `scripts/data_prep.py`: Formatting logic for LLM instruction tuning.
* `scripts/rag_builder.py`: Logic for creating and populating the ChromaDB vector store.
* `scripts/chefai.py`: **The Brain.** Main agent class that loads models and handles routing.
* `scripts/chef_tools.py`: **The Tools.** RAG search utilities for the agent.
* `scripts/api.py`: The Backend Server (FastAPI).
* `scripts/frontend.py`: The Chat Interface (Streamlit).

---

## ⚙️ Setup & Configuration

### 1. Installation

Clone the repository and install dependencies:

~~~bash
git clone https://github.com/yourusername/ChefAI.git
cd ChefAI
pip install -r requirements.txt
~~~

---

### 2. ⚠️ Important: Configure the Model

Because the fine-tuned model weights are too large to host on GitHub, the system must download them from Hugging Face.

1. Open `scripts/chefai.py`
2. Locate the `__init__` function (around line 15)
3. Update the `self.chef_path` variable as shown below

~~~python
# scripts/chefai.py

# CHANGE THIS LINE:
# self.chef_path = os.path.join(os.path.dirname(__file__), "../models/mistral_qlora")

# TO THIS:
self.chef_path = "hazarsozer/Chef-Mistral-7B"
~~~

**Note:**  
The QLoRA adapters will be downloaded automatically the first time you run the application.

---

## 🚀 How to Run

You must run the **Brain** and the **Interface** in separate terminals.

---

### Terminal 1: Start the Backend

~~~bash
python scripts/api.py
~~~

Wait until you see:

~~~
Application startup complete
~~~

---

### Terminal 2: Start the Frontend

~~~bash
streamlit run scripts/frontend.py
~~~

The chat interface will open automatically in your browser.  
Enjoy cooking! 🥘
