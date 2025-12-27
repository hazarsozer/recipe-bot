import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from chefai import ChefAI

app = FastAPI(title="ChefAI Agent", description="Mistral+Phi3 Dual Agent")

print("🏗️ Booting up ChefAI... (This takes ~1 min)")
# Initialize the bot once when the server starts
bot = ChefAI()

class UserInput(BaseModel):
    text: str
    # If the frontend doesn't send one, default to "default"
    session_id: str = "default" 

@app.get("/")
def home():
    return {"status": "Online", "message": "ChefAI is ready to cook!"}

@app.post("/chat")
def chat_endpoint(request: UserInput):
    try:
        # Pass the unique ticket number (session_id) to the router
        response = bot.router(request.text, request.session_id)
        return {"response": response}
    except Exception as e:
        # Print error to terminal for debugging
        print(f"❌ API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)