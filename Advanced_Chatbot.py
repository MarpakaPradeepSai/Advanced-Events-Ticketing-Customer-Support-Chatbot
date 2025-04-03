import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time

# Define constants for model files
GITHUB_MODEL_URL = "https://github.com/your-repo/distilgpt2-model/raw/main"
MODEL_FILES = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]

def download_model_files(model_dir="/tmp/DistilGPT2_Model"):
    os.makedirs(model_dir, exist_ok=True)
    for filename in MODEL_FILES:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(local_path):
            response = requests.get(url)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
            else:
                st.error(f"Failed to download {filename} from GitHub.")
                return False
    return True

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_trf")

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Check your internet connection or GitHub URL.")
        return None, None
    model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Load models before rendering the UI
with st.spinner("Loading models, please wait..."):
    nlp = load_spacy_model()
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        st.error("Failed to load the model.")
        st.stop()

# CSS for styling
st.markdown("""
    <style>
        .stButton > button {
            background-color: green;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Manage chat state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

# Disclaimer and continue button
if not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb;">
            <h1 style="font-size: 36px;">⚠️ Disclaimer</h1>
            <p>
                This <b>Chatbot</b> assists with ticketing-related inquiries. It’s been fine-tuned on a set of intents and might not respond accurately to all queries.
            </p>
            <p>
                Intents include: Cancel Ticket, Buy Ticket, Sell Ticket, Transfer Ticket, Upgrade Ticket, Find Ticket, Change Personal Details, Get Refund, Find Upcoming Events, Customer Service, and more.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Continue", key="continue_button"):
        st.session_state.show_chat = True
        st.experimental_rerun()

# Chat interface
if st.session_state.show_chat:
    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")
    user_input = st.text_input("Your message:")
    if user_input:
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        response = model.generate(inputs, max_length=100, num_return_sequences=1)
        decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)
        st.write(f"Chatbot: {decoded_response}")
