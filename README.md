# 🍳 Constraint-Aware Recipe Generation: A Comparative LLM Fine-Tuning Study

**Course:** YZV302E Deep Learning - Istanbul Technical University  
**Authors:** * **Hazar Utku Sözer** (150220754) - Data Engineering & Data Pipeline & Deployment  
* **Faruk Rıza Öz** (150220753) - Data Engineering & Data Pipeline & Deployment

---

## 📖 Project Overview
This project aims to develop an intelligent **"Recipe-Bot"** by fine-tuning Large Language Models (LLMs) to generate personalized recipes.

**The Core Problem:** Standard LLMs (like GPT-4 or base Llama) struggle with strict constraints. They often:
1.  **Hallucinate Ingredients:** Suggesting items the user doesn't have (e.g., adding milk when the user only listed eggs).
2.  **Ignore Context:** Suggesting dinner recipes for breakfast.

**Our Solution:** We treat recipe generation as a **Constraint Satisfaction Problem**. Instead of using RAG (Retrieval-Augmented Generation), we fine-tune the model to inherently understand and respect "Constraint Contexts" (Meal Type, Dietary Restrictions, Cooking Method).

---

## 🔬 Methodology: The Comparative Approach
As outlined in our research proposal, this project compares two distinct fine-tuning strategies to determine the optimal balance between model size and training precision.

### 🛤️ Track 1: Large-Model PEFT (Current Focus)
* **Model:** `Mistral-7B-Instruct-v0.3`
* **Technique:** **QLoRA** (Quantized Low-Rank Adaptation)
* **Precision:** 4-bit (NF4) quantization with `bitsandbytes`.
* **Hardware Target:** Consumer GPU (NVIDIA RTX 4070 Super / 12GB VRAM).
* **Goal:** Efficient fine-tuning on local hardware.

### 🛤️ Track 2: Small-Model Full Fine-Tune (Planned)
* **Model:** `Phi-3-mini` (3.8B)
* **Technique:** Full Parameter Fine-Tuning.
* **Precision:** BF16 (Bfloat16).
* **Hardware Target:** Cloud/Data Center GPU.
* **Goal:** Testing if a fully trained smaller model outperforms a quantized larger model.

---

## 📂 Project Structure

```bash
├── data/
│   ├── food-com-recipes.../ # Raw Public Datasets (Kaggle/Food.com)
│   ├── processed/           # Cleaned DataFrames (Physics & Structure filtered)
│   └── training/            # JSONL files formatted for SFTTrainer
├── notebooks/
│   ├── data_gathering.ipynb # Dataset download & merging strategy
│   ├── eda.ipynb            # Exploratory Data Analysis & Cleaning Pipeline
│   └── llm_data_prep.ipynb  # Feature Engineering (Tag -> Context Mapping)
├── scripts/
│   ├── preprocessing.py     # Data cleaning logic (Structure, Calorie/Mass checks)
│   ├── data_prep.py         # Prompt engineering & Constraint extraction
│   ├── scraper.py           # Web scraper for supplementary recipes
│   └── train.py             # QLoRA Training Script (Track 1 implementation)
└── README.md
```

## 🧠 Feature Engineering: "Constraint Injection"
We transform raw dataset tags into explicit natural language constraints to prevent context hallucinations. This ensures the model treats user preferences (like "Vegan" or "Slow Cooker") as strict rules rather than optional suggestions.

| Raw Tags | Derived Context (Input to LLM) |
| :--- | :--- |
| `['breakfast', 'vegan']` | **"Context: Breakfast, Vegan"** |
| `['slow-cooker', 'beef', 'dinner']` | **"Context: Dinner, Slow Cooker"** |
| `['5-minutes', 'snack']` | **"Context: Snack, Very Quick"** |
| `['air-fryer', 'chicken']` | **"Context: General Dish, Air Fryer"** |

*See `scripts/data_prep.py` for the full mapping logic.*

---

## 🛠️ Setup & Usage

### 1. Environment Setup (Linux/WSL2)
This project requires a GPU-accelerated environment (NVIDIA Drivers required).
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
# Key deps: torch, transformers, peft, bitsandbytes, trl, accelerate
```

### 2. Data Pipeline
1. **Clean the Data:** Run `notebooks/eda.ipynb`. This merges the Kaggle/Scraped data and removes "physically impossible" recipes (e.g., recipes with >5000 calories or 0g mass) and structurally broken rows.
2. **Generate Training Prompts:** Run `notebooks/llm_data_prep.ipynb`. This applies the constraint mapping (Meal, Diet, Cooking Method) and exports the dataset to `data/training/llm_train.jsonl`.

### 3. Training 
To be filled.

### 4. Inference/Chatbot (Deployment)
**Planned Phase:** The final model will be served via a FastAPI backend. To enable the "Continuous Chatbot" feature, we will use a Redis cache to manage conversational history (e.g., handling follow-up constraints like "I also have cheese"), separating the "Reasoning Brain" (Mistral) from the "Conversation Memory" (App Layer).