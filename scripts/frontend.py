import streamlit as st
import requests
import uuid

# CONFIGURATION
API_URL = "http://localhost:8000/chat"
st.set_page_config(page_title="ChefAI", page_icon="👨‍🍳", layout="centered")

# SESSION ID GENERATION (The Ticket Number)
# This runs once per user when they open the page
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    # Print to frontend terminal to confirm unique IDs are generating
    print(f"🎫 New Session Started: {st.session_state.session_id}")

# UI HEADER 
st.title("👨‍🍳 ChefAI Assistant")
st.markdown("I am your personalized cooking assistant. Tell me what ingredients you have!")

# SESSION STATE (Memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

# DISPLAY CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# CHAT INPUT
if prompt := st.chat_input("What's in your fridge?"):
    # Display User Message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Fetch Response from Backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send request to your local API WITH the Session ID
                payload = {
                    "text": prompt,
                    "session_id": st.session_state.session_id
                }
                
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    bot_reply = response.json().get("response", "Error: No response key.")
                else:
                    bot_reply = f"Error {response.status_code}: {response.text}"
            
            except requests.exceptions.ConnectionError:
                bot_reply = "🚨 Error: Could not connect to ChefAI backend. Is 'api.py' running?"

        st.write(bot_reply)
    
    # Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})