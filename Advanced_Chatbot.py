import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# --- Configuration ---
GITHUB_REPO_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot"
MODEL_DIR_NAME = "DistilGPT2_Model"

MODEL_PATH = os.path.join(MODEL_DIR_NAME) # For local testing if needed
TOKENIZER_PATH = os.path.join(MODEL_DIR_NAME) # For local testing if needed

# --- Function to load model and tokenizer from GitHub ---
@st.cache_resource  # Use cache_resource for efficient loading
def load_model_and_tokenizer_from_github(github_repo_url, model_dir_name):
    model_url = f"{github_repo_url}/raw/main/{model_dir_name}"
    tokenizer = GPT2Tokenizer.from_pretrained(model_url)
    model = GPT2LMHeadModel.from_pretrained(model_url)
    return tokenizer, model

# --- Load Model and Tokenizer ---
tokenizer, model = load_model_and_tokenizer_from_github(GITHUB_REPO_URL, MODEL_DIR_NAME)
device = model.device  # Get the model's device (CPU in Streamlit Cloud)

# --- Response Generation Function ---
def generate_response(instruction, max_length=256):
    model.eval()
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
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

# --- Streamlit UI ---
st.title("Event Ticketing Support Chatbot")

user_input = st.text_input("Enter your question or instruction:", "")

if user_input:
    processed_input = user_input[0].upper() + user_input[1:] # Capitalize first letter
    with st.spinner("Generating response..."):
        response = generate_response(processed_input)
    st.text_area("Chatbot Response:", value=response, height=200)
