import torch
from unsloth import FastLanguageModel
from chef_tools import ChefTools
import os
import textwrap

class ChefAI:
    def __init__(self):
        print("🤖 Initializing ChefAI...")

        # Initializing tools
        self.tools = ChefTools()

        # Model paths
        self.chef_path = os.path.join(os.path.dirname(__file__), "../models/mistral_qlora")
        self.waiter_name = "unsloth/Phi-3-mini-4k-instruct"

        # Loading main model
        print("🍳 Loading Chef model...")
        self.chef_model, self.chef_tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.chef_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.chef_model)

        # Loading waiter model
        print("🧑‍🍳 Loading Waiter model...")
        self.waiter_model, self.waiter_tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.waiter_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.waiter_model)

        # Session state
        self.current_dish = None
        self.current_recipe_text = None
        self.chat_history = []

        print("✅ ChefAI is ready to serve!")

    def run_inference(self, model, tokenizer, prompt, max_tokens=512, repeat_penalty=1.1, temperature=0.6):
        """ 
        Run inference on the given model with the provided prompt. 
        """

        # Tokenize input
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repeat_penalty,
            use_cache=True,
            temperature = temperature,
            do_sample = True
        )

        # Decode output
        new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]

        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    def _update_history(self, user_text, bot_text):
        """
        Internal helper to save chat to memory.
        """

        # Appending new interaction
        self.chat_history.append(f"User: {user_text}")
        self.chat_history.append(f"Chef: {bot_text}")

        # Enforcing sliding window
        if len(self.chat_history) > 6:
            self.chat_history = self.chat_history[-6:]
    
    def router(self, user_input):
        """
        Route the user input to the appropriate model (Brain or Mouth).
        """
        
        # Classification prompt
        prompt = textwrap.dedent(f"""<|user|>
        Task: You're a logic router. Classify the User Input into exactly one category: RECIPE, CHAT, or FOOD_RELATED.

        DEFINITIONS:
        1. RECIPE: User wants to cook, eat, or asks for a dish suggestion. (e.g. "What can I make?", "I have chicken").
        2. CHAT: User is greeting, introducing themselves, or stating their identity. (e.g. "Hi", "Who are you?", "I love food").
        3. FOOD_RELATED: User asks about science, safety, or measurements. (e.g. "Is this safe?", "How many grams?").

        User Input: "{user_input}"
        
        OUTPUT ONLY THE CATEGORY NAME.<|end|>
        <|assistant|>""").strip()

        intent = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=10,
            temperature=0.1
        ).upper()

        intent = intent.replace("CATEGORY:", "").strip().split()[0].replace(".", "")

        if "RECIPE" in intent:
            return self.handle_recipe(user_input)
        elif "CHAT" in intent:
            return self.handle_chat(user_input)
        elif "FOOD_RELATED" in intent:
            return self.handle_food_related(user_input)
        else:
            # If router gets confused default to CHAT
            return self.handle_chat(user_input)
        
    def handle_chat(self, user_input):
        """
        Gives the output if intent is to chat.
        """

        # History string
        history_str = '\n'.join(self.chat_history)

        prompt = textwrap.dedent(f"""<|user|>
        Task: You're the front model of an AI Chef Agent. When user comes with intent 
        to chat, you do the talking politely like a host greeting and chatting with 
        his customers. You should try to comply with users requests, but if the topic
        is too far away from culinary you should remind them you're a Chef here to 
        help them about their culinary questions. Usually try to keep it short.

        PREVIOUS CONVERSATION:
        {history_str}

        CURRENT USER INPUT: "{user_input}"
        <|end|><|assistant|>""").strip()

        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=150
        )

        self._update_history(user_input, output)

        return output
    
    def handle_food_related(self, user_input):
        """
        Step 1: Sub-classify the intent.
        Step 2: Route to the correct tool or model.
        """

        # Sub-routing logic
        prompt = textwrap.dedent(f"""<|user|>
        User Input: "{user_input}"

        Task: Classify this specific food related question or request.
        - "SAFETY": If user input is about food safety, hygiene, storage and dangerous items or if the input user gives can cause safety hazards.
        - "CONSTANTS": If user input is about unit conversion (imperial/metric), substitutions for ingredients, nutrition macros, or specific ingredient weights or nutrition tables.
        - "INSTRUCT": If user input is confused about a step in recipe, thinks that it's vague and needs more detail on a technique, or asks a question such as "how do I do step 5?" etc.
        - "ELSE": If user input is doesn't fit onto any categories on the above, and more vague topics like food history, science or complex food theories etc.

        OUTPUT ONLY ONE CATEGORY NAME: "SAFETY", "CONSTANTS", "INSTRUCT", or "ELSE".<|end|>
        <|assistant|>""").strip()

        sub_intent = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=10
        ).upper()

        # Routing
        if "SAFETY" in sub_intent:
            return self.handle_safety_query(user_input)
        
        elif "CONSTANTS" in sub_intent:
            return self.handle_constant_query(user_input)
        
        elif "INSTRUCT" in sub_intent:
            return self.handle_instruction_query(user_input)
        
        else:
            return self.escalate_to_brain(user_input)
        
    def handle_safety_query(self, user_input):
        """
        Specialist function for safety questions.
        Uses RAG to find specific rules in 'chef_safety' collection.
        """

        safety_context = self.tools.check_safety(user_input)
        if not safety_context:
            safety_context = "No specific strict rules found. Use your general food safety knowledge."

        recipe_context = ""
        if self.current_recipe_text:
            recipe_context = f"\nCURRENT RECIPE STEPS:\n{self.current_recipe_text}\n"

        # Specialist prompt
        prompt = textwrap.dedent(f"""<|user|>
        User Input: "{user_input}"

        {recipe_context}

        OFFICIAL SAFETY GUIDELINES FOR YOUR USE:
        {safety_context}

        Task: You're the front model of an AI Chef Agent. As a Chef Instructor, answer
        the user's question or request about safety politely and strictly based on 
        the GUIDELINES that given above. If the user refers to a specific step (e.g. "Is step 5 safe?"),
        USE THE CURRENT RECIPE STEPS above to verify. If the guidelines don't cover it, user your
        general knowledge but be extremely cautious. 
        Start with "⚠️ SAFETY FIRST:" if there's a risk.<|end|>
        <|assistant|>""").strip()

        # Output generation
        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=200
        )

        return output
    
    def handle_constant_query(self, user_input):
        """
        Specialist function for culinary conversion and substitution questions.
        Uses RAG to find specific rules in culinary_constans JSON file.
        """

        # Get data context for conversion or substitutions
        data_context = self.tools.search_constants(user_input)

        if not data_context:
            data_context = "No specific context found about the user input. Use your general knowledge in culinary."
        
        # Specialist prompt
        prompt = textwrap.dedent(f"""<|user|>
        User Input: "{user_input}"

        DATA FOUND ABOUT CONTEXT FOR YOUR USE:
        {data_context}

        Task: You're the front model of an AI Chef Agent. As a Chef Instructor, you should
        answer user's question or request politely using the DATA that given above. 
        Be precise with your numbers. If the data don't cover it, use your general 
        knowledge about conversions, substitutions or other type of knowledge to give
        an answer to user.<|end|>
        <|assistant|>""").strip()

        # Output generation
        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=100,
            temperature=0.1 # Forcing strict logic for math
        )

        return output
    
    def handle_instruction_query(self, user_input):
        """
        Speciaist function for detailing and explaining instructions.
        Uses 'self.current_dish' to give context-aware advice.
        """

        # Check if we have the specific instructions in memory
        if self.current_recipe_text:
            context_str = f"USER IS COOKING: '{self.current_dish}'\n\nOFFICIAL RECIPE STEPS:\n{self.current_recipe_text}"
        elif self.current_dish:
            context_str = f"USER IS COOKING: '{self.current_dish}' (No specific steps loaded)."
        else:
            context_str = "User is asking a general cooking question."

        # Specialist prompt
        prompt = textwrap.dedent(f"""<|user|>
        User Input: "{user_input}"

        CONTEXT: {context_str}

        Task: You're the front model of an AI Chef Agent. As a Chef Instructor, you should
        answer user's request or confusion like a teacher. If user asks questions such as: 
        "How long?", "How much?" or explanation for a specific instruction step check the
        CONTEXT first before answering. Be patient and detailed. <|end|>
        <|assistant|>""").strip()

        # Output generation
        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=256,
        )

        return output
    
    def escalate_to_brain(self, user_input):
        """
        Step 1: Using fine-tuned Brain model for questions that Front model can't answer
        Step 2: Using front model to rephrase the explanation and give output.
        """

        # Specialist prompt
        prompt_for_brain = textwrap.dedent(f"""### Instruction:
        User Input: "{user_input}"

        Task: You're the backend Brain model of an AI Chef Agent. As the Executive chef,
        provide a detailed explanation for the user's query.

        ### Response:
        """).strip()

        # Brain model output generation
        explanation = self.run_inference(
            model=self.chef_model,
            tokenizer=self.chef_tokenizer,
            prompt=prompt_for_brain,
            max_tokens=400,
            temperature=0.3
        )

        prompt_for_mouth = textwrap.dedent(f"""<|user|>
        User Input: "{user_input}"

        Explanation from backend Brain Model: 
        {explanation}

        Task: You're the front model of an AI Chef Agent. Use the explanation provided
        above from the backend finetuned model, rephrase it if needed to answer the user's
        question.<|end|>
        <|assistant|>""").strip()

        # Output to user
        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt_for_mouth,
            max_tokens=200
        )

        return output
    
    def handle_recipe(self, user_input):
        """
        Workflow:
        -> Brain generates a dish name according to input.
        -> We sanitize it using Phi-3 to extract just the name.
        -> Using retrieval tools we check if we have any matching recipes.
        -> Brain generates the recipe using retrieved recipes.
        -> Front model presents it nicely (with repeat_penalty disabled).
        """

        # --- STEP 1: IDEATION (Mistral) ---
        ideation_prompt = textwrap.dedent(f"""### Instruction:
        Task: You're the backend Brain model of an AI Chef Agent. As an Executive
        Chef, identify the dish name based on the user's request. Output ONLY the name.

        Input: I have beef and potatoes.
        Example Dish Name: Beef Stew

        Input: Make me something with eggs.
        Example Dish Name: Omelet

        Input: I have salmon and rice.
        Example Dish Name: Pan Seared Salmon

        Input: "{user_input}"
        Dish Name:
        ### Response:
        """).strip()

        raw_idea = self.run_inference(
            model=self.chef_model,
            tokenizer=self.chef_tokenizer,
            prompt=ideation_prompt,
            max_tokens=50
        ).strip('"').split('\n')[0]

        # --- STEP 2: SANITIZATION (Phi-3) ---
        extraction_prompt = textwrap.dedent(f"""<|user|>
        Task: Extract the exact food dish name from the text below. 
        Remove any filler words like "I suggest", "Try", "Dish:", punctuation, or recipes.
        Output ONLY the dish name.

        Text: "{raw_idea}"
        <|end|><|assistant|>""").strip()

        clean_idea = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=extraction_prompt,
            max_tokens=20,
            temperature=0.1
        )

        self.current_dish = clean_idea

        # --- STEP 3: RETRIEVAL ---
        dish_recipes = self.tools.get_recipes(clean_idea)
        
        if not dish_recipes:
            dish_recipes = "No specific recipes for reference. Use your own knowledge and training."

        # --- STEP 4: GENERATION (Mistral) ---
        recipe_prompt = textwrap.dedent(f"""### Instruction:
        User Input: "{user_input}"
        Dish Name: {clean_idea}

        REFERENCES:
        {dish_recipes}

        Task: You're the backend Brain model of an AI Chef Agent. As the Executive chef,
        use the references above that given to you to create or suggest a recipe to the
        user. If no specific references given, use your own knowledge and training to come
        up with a recipe for user's needs.

        ### Response:
        """).strip()

        recipe = self.run_inference(
            model=self.chef_model,
            tokenizer=self.chef_tokenizer,
            prompt=recipe_prompt,
            max_tokens=600,
            temperature=0.3
        ) 

        # For safety
        if len(recipe) > 2500:
            recipe = recipe[:2500] + "... (truncated)"

        self.current_recipe_text = recipe

        # --- STEP 5: PLATING (Phi-3) ---
        plating_prompt = textwrap.dedent(f"""<|user|>
        User Input: "{user_input}"
        Recipe: 
        {recipe}

        Task: You are the front model of an AI Chef Agent. As a Chef Instructor,
        rewrite the recipe from Executive Chef politely.<|end|>
        <|assistant|>""").strip()

        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=plating_prompt,
            max_tokens=600,
            repeat_penalty=1.0 # CRITICAL FIX: Allows Phi-3 to rewrite naturally
        )

        return output


# Main Execution Loop
if __name__ == "__main__":
    bot = ChefAI()

    print("\n" + "="*50)
    print("👨‍🍳 CHEF AI IS READY TO SERVE!")
    print("Type 'exit' or 'quit' to close the kitchen.")
    print("="*50 + "\n")

    while True:
        try:
            # User input
            user_input = input("You: ").strip()

            # Exit check
            if user_input.lower() in ["exit", "quit"]:
                print("👨‍🍳 ChefAI: Kitchen is closing. Bon appétit!")
                break

            # Skipping empty inputs
            if not user_input:
                continue

            # The router handles everything
            print("\n... 👨‍🍳 Chef is thinking ...\n")
            response = bot.router(user_input)

            # Print response
            print(f"\n👨‍🍳 ChefAI: {response}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n👨‍🍳 ChefAI: Force shutdown detected. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
