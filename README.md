# 👨‍🍳 ChefAI: The Constraint-Aware Cooking Agent

**ChefAI** is a smart cooking assistant that actually listens to your constraints.

Unlike standard AI that might suggest *“Cheeseburgers”* when you ask for a *“Vegan Dinner,”* ChefAI uses a **Dual-Agent Architecture** to separate creativity from logic. It runs locally on your computer, ensuring privacy and strict adherence to your inventory.

---

## 📂 Project Structure (What is everything?)

Here is a quick guide to understanding the code in this repository:

### 1. The Notebooks (The Research Phase)

- `notebooks/1_data_gathering.ipynb`  
  Scrapes recipes from the web and downloads datasets. It specifically looks for structured data (such as prep time and nutrition).

- `notebooks/2_eda.ipynb`  
  The **Physics filter**. It calculates the caloric density of recipes and removes hallucinated or broken data (e.g., zero-calorie meals).

- `notebooks/3_training.ipynb`  
  The training code. This is where **Mistral 7B** was fine-tuned using QLoRA to create the *Chef* model.

- `notebooks/4_evaluation.ipynb`  
  The test bench. Models were evaluated against 100 *Hallucination Traps* (e.g., asking for vegan eggs) to demonstrate improved safety over standard models.

---

### 2. The Scripts (The Application Phase)

- `scripts/chefai.py`  
  **The Brain.** Loads the models and decides whether the user wants to chat, cook, or ask safety-related questions.

- `scripts/chef_tools.py`  
  **The Tools.** Handles the Vector Database (RAG) for safety rules and recipe retrieval.

- `scripts/api.py`  
  Backend server implemented with FastAPI.

- `scripts/frontend.py`  
  Chat interface built with Streamlit.

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
