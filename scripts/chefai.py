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

        # Multi-user session storage
        self.sessions = {}

        print("✅ ChefAI is ready to serve!")

    def _get_session(self, session_id):
        """
        Retrieves the specific memory for a user. 
        If it doesn't exist, creates a new empty memory slot.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "current_dish": None,
                "current_recipe_text": None,
                "chat_history": []
            }
        return self.sessions[session_id]

    def run_inference(self, model, tokenizer, prompt, max_tokens=512, repeat_penalty=1.1, temperature=0.6):
        """ 
        Run inference on the given model with the provided prompt. 
        """
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repeat_penalty,
            use_cache=True,
            temperature = temperature,
            do_sample = True
        )
        new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    def _update_history(self, session_id, user_text, bot_text):
        """
        Internal helper to save chat to SPECIFIC USER memory.
        """
        session = self._get_session(session_id)
        session['chat_history'].append(f"User: {user_text}")
        session['chat_history'].append(f"Chef: {bot_text}")

        # Enforcing sliding window
        if len(session['chat_history']) > 6:
            session['chat_history'] = session['chat_history'][-6:]
    
    def router(self, user_input, session_id="default"):
        """
        Route the user input to the appropriate model (Brain or Mouth).
        Now accepts session_id to track context.
        """
        
        # Classification prompt
        prompt = textwrap.dedent(f"""<|user|>
        Task: Classify User Input into ONE category: RECIPE, CHAT, or FOOD_RELATED.

        DEFINITIONS:
        1. RECIPE: User explicitly asks for a NEW dish, a different dish, or provides NEW ingredients to start over. (e.g. "Cook X", "No, I want Salmon", "I have beef instead").
        2. CHAT: Greetings, compliments, or general conversation. (e.g. "Hi", "Thanks", "Good evening").
        3. FOOD_RELATED: Questions about the CURRENT recipe. Includes: Steps, Safety, Scaling portions, or asking for measurements. (e.g. "How much?", "For 3 people?", "Next step", "Is it safe?").

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

        # DEBUG START
        print(f"🔀 [DEBUG][{session_id}] Router Intent: {intent}")
        # DEBUG END

        if "RECIPE" in intent:
            return self.handle_recipe(user_input, session_id)
        elif "CHAT" in intent:
            return self.handle_chat(user_input, session_id)
        elif "FOOD_RELATED" in intent:
            return self.handle_food_related(user_input, session_id)
        else:
            return self.handle_chat(user_input, session_id)
        
    def handle_chat(self, user_input, session_id):
        """
        Gives the output if intent is to chat.
        """
        session = self._get_session(session_id)
        
        # History string from SESSION storage
        history_str = '\n'.join(session['chat_history'])

        prompt = textwrap.dedent(f"""<|user|>
        Task: You are the AI Chef.
        
        RULES:
        1. Be warm and friendly, but EXTREMELY CONCISE.
        2. Answer in 1 or 2 sentences MAXIMUM if a long response is not necessary.
        3. Do not give long speeches.
        4. If the user says "Thank you", reply with "You're welcome! Enjoy your meal." or "Bon Appétit!".
        5. If greeting, just say hello and ask what they want to cook.

        PREVIOUS CONVERSATION:
        {history_str}

        CURRENT USER INPUT: "{user_input}"
        <|end|><|assistant|>""").strip()

        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=100,
            temperature=0.3
        )

        self._update_history(session_id, user_input, output)

        return output
    
    def handle_food_related(self, user_input, session_id):
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

        # DEBUG START
        print(f"🚦 [DEBUG][{session_id}] Food Sub-Intent: {sub_intent}")
        # DEBUG END

        # Routing
        if "SAFETY" in sub_intent:
            return self.handle_safety_query(user_input, session_id)
        elif "CONSTANTS" in sub_intent:
            return self.handle_constant_query(user_input, session_id)
        elif "INSTRUCT" in sub_intent:
            return self.handle_instruction_query(user_input, session_id)
        else:
            return self.escalate_to_brain(user_input)
        
    def handle_safety_query(self, user_input, session_id):
        """
        Specialist function for safety questions.
        Uses RAG to find specific rules in 'chef_safety' collection.
        """
        session = self._get_session(session_id)
        
        safety_context = self.tools.check_safety(user_input)
        
        # DEBUG START
        print(f"🛡️ [DEBUG] RAG Safety Context: {str(safety_context)[:100]}...")
        # DEBUG END

        if not safety_context:
            safety_context = "No specific strict rules found. Use your general food safety knowledge."

        recipe_context = ""
        # Accessing session specific recipe text
        if session['current_recipe_text']:
            recipe_context = f"\nCURRENT RECIPE STEPS:\n{session['current_recipe_text']}\n"

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

        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=200
        )

        return output
    
    def handle_constant_query(self, user_input, session_id):
        """
        Specialist function for culinary conversion and substitution questions.
        Now context-aware!
        """
        session = self._get_session(session_id)
        
        # Load the current recipe context if it exists
        if session['current_recipe_text']:
            recipe_context = f"\nCURRENT RECIPE CONTEXT:\n{session['current_recipe_text']}\n"
        else:
            recipe_context = ""

        # Get data context for conversion or substitutions
        data_context = self.tools.search_constants(user_input)

        # DEBUG START
        print(f"📏 [DEBUG][{session_id}] RAG Constants Data: {str(data_context)[:100]}...")
        # DEBUG END

        if not data_context:
            data_context = "No specific context found about the user input. Use your general knowledge in culinary."
        
        prompt = textwrap.dedent(f"""<|user|>
        User Input: "{user_input}"
        
        {recipe_context}

        REFERENCE DATA (May be irrelevant):
        {data_context}

        Task: You are a helpful Chef Instructor. Answer the user's request.
        
        RULES:
        1. SCALING: If the user asks to scale (e.g. "for 4 people"), FIRST check the ingredients to guess how many the recipe ALREADY serves. Only increase amounts if necessary.
        2. SUBSTITUTIONS: If the user asks to replace an ingredient (due to allergy or missing item), suggest a simple, common culinary alternative (e.g. "Use Vegetable Oil or Canola Oil").
        3. INTELLIGENCE: Use the REFERENCE DATA *only* if it directly answers the question. If the data talks about "Smoke Points" but the user asks about "Allergies", IGNORE THE DATA and use your own knowledge.
        4. SCALING: If scaling, use the CURRENT RECIPE CONTEXT to calculate numbers.
        5. Keep it short and helpful.
        <|end|>
        <|assistant|>""").strip()

        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=200,
            temperature=0.1
        )

        return output
    
    def handle_instruction_query(self, user_input, session_id):
        """
        Specialist function for detailing and explaining instructions.
        Uses SESSION specific context.
        """
        session = self._get_session(session_id)
        current_dish = session['current_dish']
        
        # Sanitize the dish name variable
        if current_dish and "\n" in current_dish:
            safe_dish_name = current_dish.split('\n')[0]
        else:
            safe_dish_name = current_dish

        # Check if we have the specific instructions in memory
        if session['current_recipe_text']:
            context_str = f"CURRENT RECIPE:\n{session['current_recipe_text']}"
        elif safe_dish_name:
            context_str = f"USER IS COOKING: '{safe_dish_name}'"
        else:
            context_str = "No specific recipe loaded."

        prompt = textwrap.dedent(f"""<|user|>
        CONTEXT: 
        {context_str}

        User Input: "{user_input}"

        Task: Answer the user's question based ONLY on the context.
        RULES:
        1. Ignore any weird formatting in the context.
        2. If asking for the First Step, look at Step 1 in the recipe.
        3. If asking for quantities (e.g. "for 3 people"), ESTIMATE based on the ingredients.
        4. Keep it simple and direct.
        <|end|>
        <|assistant|>""").strip()

        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=256,
            temperature=0.1 
        )

        return output
    
    def escalate_to_brain(self, user_input):
        """
        General explanations don't usually require specific session context.
        """
        prompt_for_brain = textwrap.dedent(f"""### Instruction:
        User Input: "{user_input}"

        Task: You're the backend Brain model of an AI Chef Agent. As the Executive chef,
        provide a detailed explanation for the user's query.

        ### Response:
        """).strip()

        explanation = self.run_inference(
            model=self.chef_model,
            tokenizer=self.chef_tokenizer,
            prompt=prompt_for_brain,
            max_tokens=400,
            temperature=0.3
        )

        # DEBUG START
        print(f"🧠 [DEBUG] Brain Escalation Response: {explanation[:150]}...")
        # DEBUG END

        prompt_for_mouth = textwrap.dedent(f"""<|user|>
        User Input: "{user_input}"

        Explanation from backend Brain Model: 
        {explanation}

        Task: You're the front model of an AI Chef Agent. Use the explanation provided
        above from the backend finetuned model, rephrase it if needed to answer the user's
        question.<|end|>
        <|assistant|>""").strip()

        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt_for_mouth,
            max_tokens=200
        )

        return output
    
    def handle_recipe(self, user_input, session_id):
        """
        Workflow with SESSION ID support.
        """
        session = self._get_session(session_id)
        current_dish = session.get('current_dish')
        
        print(f"\n🔎 [DEBUG][{session_id}] --- Starting Recipe Pipeline ---")

        # Ideation step (Mistral)
        if current_dish:
             print(f"🧠 [DEBUG][{session_id}] Context Found: '{current_dish}'")
             ideation_prompt = textwrap.dedent(f"""
             Context: The user is currently cooking "{current_dish}".
             
             Input: Make it spicy.
             Dish: Spicy {current_dish}
             
             Input: Remove the vegetables.
             Dish: {current_dish} (Meat Only)

             Input: I want something else.
             Dish: [New Dish Name]

             Input: "{user_input}"
             Dish:""").strip()
        else:
             # Standard Cold Start Prompt
             ideation_prompt = textwrap.dedent(f"""
             Input: I have beef.
             Dish: Beef Stew
             
             Input: I want salmon.
             Dish: Pan Seared Salmon
             
             Input: "{user_input}"
             Dish:""").strip()

        raw_idea = self.run_inference(
            model=self.chef_model,
            tokenizer=self.chef_tokenizer,
            prompt=ideation_prompt,
            max_tokens=50, 
            temperature=0.1
        ).strip('"').strip()

        if "\n" in raw_idea:
            raw_idea = raw_idea.split("\n")[0]

        if not raw_idea or len(raw_idea) < 2:
            print(f"⚠️ [DEBUG][{session_id}] Mistral Brain Freeze. Using User Input.")
            raw_idea = user_input

        # DEBUG START
        print(f"🧠 [DEBUG][{session_id}] Step 1 - Mistral Raw Idea: '{raw_idea}'")
        # DEBUG END

        # Cleaning step (Phi-3)
        extraction_prompt = textwrap.dedent(f"""<|user|>
        Task: Extract the specific food dish name from the Input Text.
        
        RULES:
        1. Remove conversational fillers (e.g. "I suggest", "Here is").
        2. Keep the full dish name including adjectives (e.g. "Spicy", "Vegan").
        3. CRITICAL: Do NOT add any new words. Only use words found in the Input Text.
        4. Output ONLY the cleaned name.

        Text: "{raw_idea}"
        <|end|><|assistant|>""").strip()

        clean_idea = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=extraction_prompt,
            max_tokens=50,
            temperature=0.1
        )

        # Saving to session
        session['current_dish'] = clean_idea

        # DEBUG START
        print(f"🧼 [DEBUG][{session_id}] Step 2 - Phi-3 Cleaned Name: '{clean_idea}'")
        # DEBUG END

        # Retrieving recipes from RAG
        dish_recipes = self.tools.get_recipes(clean_idea)
        
        # DEBUG START
        if dish_recipes:
             print(f"📚 [DEBUG][{session_id}] Step 3 - RAG Found Content for '{clean_idea}'")
        else:
             print(f"📚 [DEBUG][{session_id}] Step 3 - RAG Found NOTHING")
        # DEBUG END

        if not dish_recipes:
            dish_recipes = "No specific recipes for reference. Use your own knowledge and training."

        # Recipe generation (Mistral)
        recipe_prompt = textwrap.dedent(f"""### Instruction:
        User Input: "{user_input}"
        Target Dish: {clean_idea}

        REFERENCES:
        {dish_recipes}

        Task: Write ONE single recipe for {clean_idea}.
        1. CRITICAL: PRIORITIZE THE USER INPUT INGREDIENTS.
        2. If References don't match or use different ingredients (e.g. wrong meat), IGNORE THEM and write your own recipe.
        3. Include Ingredients and Steps.
        4. DO NOT write a second recipe. STOP after the steps.

        ### Response:
        Title: {clean_idea}
        """).strip()

        recipe = self.run_inference(
            model=self.chef_model,
            tokenizer=self.chef_tokenizer,
            prompt=recipe_prompt,
            max_tokens=600,
            temperature=0.3
        )

        if len(recipe) > 2500:
            recipe = recipe[:2500] + "... (truncated)"

        # Saving to session
        session['current_recipe_text'] = recipe

        # DEBUG START
        print(f"📝 [DEBUG][{session_id}] Step 4 - Mistral Generated Recipe (First 400 chars):")
        print(f"      {recipe[:400].replace(chr(10), ' ')}...")
        # DEBUG END

        # Plating for user (Phi-3)
        plating_prompt = textwrap.dedent(f"""<|user|>
        You are a Chef Editor. Format the raw recipe below into clean, readable Markdown.
        
        RULES:
        1. Check the ingredients. If quantities are missing, ESTIMATE standard amounts for 1 PERSON (Single Serving), unless the "User Request" explicitly asks for more.
        2. Use a Bold Title.
        3. Use a Bulleted List for Ingredients (ensure every item has a quantity).
        4. Use a Numbered List for Steps.
        5. Be polite but brief in the intro.
        6. CRITICAL: If the raw text contains multiple recipes (e.g. "Option 2"), IGNORE THEM. Only format the FIRST recipe.

        RAW RECIPE: 
        {recipe}
        <|end|>
        <|assistant|>""").strip()

        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=plating_prompt,
            max_tokens=600,
            repeat_penalty=1.05, 
            temperature=0.2 
        )

        print(f"✅ [DEBUG][{session_id}] --- Pipeline Complete ---\n")

        return output

# Main Execution Loop
if __name__ == "__main__":
    bot = ChefAI()

    print("\n" + "="*50)
    print("👨‍🍳 CHEF AI IS READY TO SERVE! (CLI MODE)")
    print("Type 'exit' or 'quit' to close the kitchen.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("👨‍🍳 ChefAI: Kitchen is closing. Bon appétit!")
                break

            if not user_input:
                continue

            print("\n... 👨‍🍳 Chef is thinking ...\n")
            
            # CLI uses a hardcoded ID "cli_user"
            response = bot.router(user_input, session_id="cli_user")

            print(f"\n👨‍🍳 ChefAI: {response}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n👨‍🍳 ChefAI: Force shutdown detected. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
