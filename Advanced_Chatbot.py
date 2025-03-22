import streamlit as st
from transformers import AutoModelForCausalLM, GPT2Tokenizer
import torch
import os
import requests
from pathlib import Path

MODEL_DIR = Path("DistilGPT2_Model")
GITHUB_BASE_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model/"

# Create model directory
MODEL_DIR.mkdir(exist_ok=True)

REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "merges.txt",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json"
]

# Download with validation
for filename in REQUIRED_FILES:
    file_path = MODEL_DIR / filename
    if not file_path.exists():
        try:
            with st.spinner(f"Downloading {filename}..."):
                response = requests.get(GITHUB_BASE_URL + filename)
                response.raise_for_status()  # Validate download
                with open(file_path, "wb") as f:
                    f.write(response.content)
        except Exception as e:
            st.error(f"Failed to download {filename}: {str(e)}")
            st.stop()

@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            use_safetensors=True,
            trust_remote_code=True  # Add this for custom architectures
        ).to(device)
        
        tokenizer = GPT2Tokenizer.from_pretrained(str(MODEL_DIR))
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model, tokenizer = load_model()

# ... rest of your code ...

def generate_response(instruction, max_length=256):
    device = model.device
    input_text = f"Instruction: {instruction} Response:"
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# Streamlit interface
st.set_page_config(page_title="Ticket Support Chatbot", page_icon="🤖")
st.title("Events Ticketing Support Chatbot")

user_input = st.text_input("Ask something about ticketing:", 
                         placeholder="How do I cancel my ticket?")

if st.button("Get Response") or user_input:
    if user_input:
        user_query = user_input[0].upper() + user_input[1:]
        
        with st.spinner("Generating response..."):
            response = generate_response(user_query)
        
        st.success("Response generated!")
        st.markdown(f"**User:** {user_query}")
        st.markdown(f"**Chatbot:** {response}")
    else:
        st.warning("Please enter your query first.")

st.markdown("---")
st.markdown("Powered by DistilGPT-2 | Developed by [Your Name]")
