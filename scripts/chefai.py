import torch
from unsloth import FastLanguageModel
from chef_tools import ChefTools
import os

class ChefAI:
    def __init__(self):
        print("🤖 Initializing ChefAI...")

        # Initializing tools
        self.tools = ChefTools()

        # Model paths
        self.chef_path = os.path.join(os.path.dirname(__file__), "../models/mistral_qlora")
        self.waiter_name = "unsloth/Phi-3-mini-4k-instruct"

        #Loading main model
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
        self.last_generated_reply = None
        self.chat_history = []

        print("✅ ChefAI is ready to serve!")

    def run_inference(self, model, tokenizer, prompt, max_tokens=512, repeat_penalty=1.1, temperature=0.7):
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
    
    def router(self, user_input):
        """
        Route the user input to the appropriate model (Brain or Mouth).
        """
        
        # Classification prompt
        prompt = f"""<|user|>
        User Input: "{user_input}"

        Task: Classify the user's intent.
        - "RECIPE": If user wants to eat, lists ingredients, asks for a recipe, a menu, or if user doesn't know what to eat and wants a random suggestion, or says "yes/sure" or any kind of approving to a food offer.
        - "CHAT": If user is greeting, asking "how are you?", trying to discuss non-food topics (such as movies or weather etc.)
        - "FOOD RELATED": If user is not directly asking for a recipe or a suggestion, but instead asking a question about food safety, cooking instructions, about nutritions or meaasurements.

        OUTPUT ONLY ONE CLASSIFICATION: "RECIPE", "CHAT" or "FOOD RELATED".<|end|>
        <|assistant|>"""

        intent = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=20
        ).upper()

        if "RECIPE" in intent:
            return self.handle_recipe(user_input)
        elif "CHAT" in intent:
            return self.handle_chat(user_input)
        else:
            return self.handle_food_related(user_input)
        
    def handle_chat(self, user_input):
        """
        Gives the output if intent is to chat.
        """

        prompt = f"""<|user|>
        User Input: "{user_input}"

        Task: You're the front model of an AI Chef Agent. When user comes with intent 
        to chat, you do the talking politely like a host greeting and chatting with 
        his customers. You should try to comply with users requests, but if the topic
        is too far away from culinary you should remind them you're a Chef here to 
        help them about their culinary questions. Usually try to keep it short.<|end|><|assistant|>
        """

        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=512
        )

        return output
    
    def handle_food_related(self, user_input):
        """
        Step 1: Sub-classify the intent.
        Step 2: Route to the correct tool or model.
        """

        # Sub-routing logic
        prompt = f"""<|user|>
        User Input: "{user_input}"

        Task: Classify this specific food related question or request.
        - "SAFETY": If user input is about food safety, hygiene, storage and dangerous items or if the input user gives can cause safety hazards.
        - "CONSTANTS": If user input is about unit conversion (imperial/metric), substitutions for ingredients, nutrition macros, or specific ingredient weights or nutrition tables.
        - "INSTRUCT": If user input is confused about a step in recipe, thinks that it's vague and needs more detail on a technique, or asks a question such as "how do I do step 5?" etc.
        - "ELSE": If user input is doesn't fit onto any categories on the above, and more vague topics like food history, science or complex food theories etc.

        OUTPUT ONLY ONE: "SAFETY", "CONSTANTS", "INSTRUCT", or "ELSE".<|end|>
        <|assistant|>"""

        sub_intent = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=10
        ).upper()

        print(f"  ↳Sub-Intent identified: {sub_intent}")

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

        # Specialist prompt
        prompt = f"""<|user|>
        User Input: "{user_input}"

        OFFICIAL SAFETY GUIDELINES FOR YOUR USE:
        {safety_context}

        Task: You're the front model of an AI Chef Agent. As a Chef Instructor, answer
        the user's question or request about safety politely and strictly based on 
        the GUIDELINES that given above. If the guidelines don't cover it, user your
        general knowledge but be extremely cautious. 
        Start with "⚠️ SAFETY FIRST:" if there's a risk.<|end|>
        <|assistant|>"""

        # Output generation
        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=256
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
        prompt = f"""<|user|>
        User Input: "{user_input}"

        DATA FOUND ABOUT CONTEXT FOR YOUR USE:
        {data_context}

        Task: You're the front model of an AI Chef Agent. As a Chef Instructor, you should
        answer user's question or request politely using the DATA that given above. 
        Be precise with your numbers. If the data don't cover it, use your general 
        knowledge about conversions, substitutions or other type of knowledge to give
        an answer to user.<|end|>
        <|assistant|>"""

        # Output generation
        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=256,
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
        prompt = f"""<|user|>
        User Input: "{user_input}"

        CONTEXT: {context_str}

        Task: You're the front model of an AI Chef Agent. As a Chef Instructor, you should
        answer user's request or confusion like a teacher. If user asks questions such as: 
        "How long?", "How much?" or explanation for a specific instruction step check the
        CONTEXT first before answering. Be patient and detailed. <|end|>
        <|assistant|>"""

        # Output generation
        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt,
            max_tokens=256,
            temperature=0.5
        )

        return output
    
    def escalate_to_brain(self, user_input):
        """
        Step 1: Using fine-tuned Brain model for questions that Front model can't answer
        Step 2: Using front model to rephrase the explanation and give output.
        """

        # Specialist prompt
        prompt_for_brain = f"""<|user|>
        User Input: "{user_input}"

        Task: You're the backend Brain model of an AI Chef Agent. As the Executive chef,
        provide a detailed explanation for the user's query.<|end|>
        <|assistant|>"""

        # Brain model output generation
        explanation = self.run_inference(
            model=self.chef_model,
            tokenizer=self.chef_tokenizer,
            prompt=prompt_for_brain,
            max_tokens=512,
            temperature=0.3
        )

        prompt_for_mouth = f"""<|user|>
        User Input: "{user_input}"

        Explanation from backend Brain Model: 
        {explanation}

        Task: You're the front model of an AI Chef Agent. Use the explanation provided
        above from the backend finetuned model, rephrase it if needed to answer the user's
        question.<|end|>
        <|assistant|>"""

        # Output to user
        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=prompt_for_mouth,
            max_tokens=256
        )

        return output
    
    def handle_recipe(self, user_input):
        """
        Workflow:
        -> Brain generates a dish name according to input.
        -> Using retrieval tools we check if we have any matching recipes in our database.
        -> Brain generates the recipe using retrieved recipes as reference.
        -> Front model presents it nicely.
        """

        ideation_prompt = f"""<|user|>
        User Input: "{user_input}"

        Task: You're the backend Brain model of an AI Chef Agent. As the Executive chef,
        suggest a dish name without listing ingredients or steps, according to user input.
        JUST OUTPUT A DISH TITLE.<|end|>
        <|assistant|>"""

        idea = self.run_inference(
            model=self.chef_model,
            tokenizer=self.chef_tokenizer,
            prompt=ideation_prompt,
            max_tokens=20
        ).strip('"')

        self.current_dish = idea

        dish_recipes = self.tools.get_recipes(idea)
        if not dish_recipes:
            dish_recipes = "No specific recipes for reference. Use your own knowledge and training."

        recipe_prompt = f"""<|user|>
        User Input: "{user_input}"
        Dish Name: {idea}

        REFERENCES:
        {dish_recipes}

        Task: You're the backend Brain model of an AI Chef Agent. As the Executive chef,
        use the references above that given to you to create or suggest a recipe to the
        user. If no specific references given, use your own knowledge and training to come
        up with a recipe for user's needs.<|end|>
        <|assistant|>"""

        recipe = self.run_inference(
            model=self.chef_model,
            tokenizer=self.chef_tokenizer,
            prompt=recipe_prompt,
            max_tokens=1024
        ) 

        self.current_recipe_text = recipe

        plating_prompt = f"""<|user|>
        User Input: "{user_input}"

        Executive Chef's Recipe: 
        {recipe}

        Task: You're the front model of an AI Chef Agent. As a Chef instructor, explain
        the recipe given from the Executive Chef above to the user. Use a polite language
        towards the user. Detail or add steps if necessary, but strictly use the recipe
        that's given to you by Executive Chef.<|end|>
        <|assistant|>"""

        output = self.run_inference(
            model=self.waiter_model,
            tokenizer=self.waiter_tokenizer,
            prompt=plating_prompt,
            max_tokens=1024
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
