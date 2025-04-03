import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time

# GitHub directory containing the DistilGPT2 model files
GITHUB_MODEL_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"

# List of model files to download
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json"
]

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
    nlp = spacy.load("en_core_web_trf")
    return nlp

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Check your internet connection or GitHub URL.")
        return None, None

    model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

static_placeholders = {
    # ... (keep existing static placeholders the same)
}

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    # ... (keep existing replacement function the same)

def extract_dynamic_placeholders(user_question, nlp):
    # ... (keep existing NER function the same)

def generate_response(model, tokenizer, instruction, max_length=256):
    # ... (keep existing generation function the same)

# Enhanced CSS with stop button styling
st.markdown(
    """
<style>
    /* Existing styles remain the same */
    
    /* Spinner and stop button styles */
    .generating-container {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .spinner {
        border: 3px solid #f3f3f3;
        border-radius: 50%;
        border-top: 3px solid #3498db;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stop-button {
        cursor: pointer;
        color: #ff4b4b;
        margin-left: auto;
        transition: transform 0.2s ease;
        background: none;
        border: none;
        padding: 0;
    }
    
    .stop-button:hover {
        transform: scale(1.2);
    }
</style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI setup remains the same until the response generation part...

# Modified response generation blocks
if process_query_button:
    if selected_query == "Choose your question":
        st.error("⚠️ Please select your question from the dropdown.")
    elif selected_query:
        prompt_from_dropdown = selected_query
        prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

        st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})
        if last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt_from_dropdown, unsafe_allow_html=True)
        last_role = "user"

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            generating_placeholder = st.empty()
            
            # Show custom generating indicator with stop button
            generating_html = """
            <div class="generating-container">
                <div class="spinner"></div>
                <span style="color: #1a1a1a;">Generating response...</span>
                <button class="stop-button">⏹️</button>
            </div>
            """
            generating_placeholder.markdown(generating_html, unsafe_allow_html=True)
            
            # Generate response
            dynamic_placeholders = extract_dynamic_placeholders(prompt_from_dropdown, nlp)
            response_gpt = generate_response(model, tokenizer, prompt_from_dropdown)
            
            # Remove generating indicator
            generating_placeholder.empty()
            
            full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        last_role = "assistant"

# Similar modification for the chat input block
if prompt := st.chat_input("Enter your own question:"):
    prompt = prompt[0].upper() + prompt[1:] if prompt else prompt
    if not prompt.strip():
        st.toast("⚠️ Please enter a question.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        if last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
        last_role = "user"

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            generating_placeholder = st.empty()
            
            generating_html = """
            <div class="generating-container">
                <div class="spinner"></div>
                <span style="color: #1a1a1a;">Generating response...</span>
                <button class="stop-button">⏹️</button>
            </div>
            """
            generating_placeholder.markdown(generating_html, unsafe_allow_html=True)
            
            dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
            response_gpt = generate_response(model, tokenizer, prompt)
            
            generating_placeholder.empty()
            
            full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        last_role = "assistant"

    # Conditionally display reset button
    if st.session_state.chat_history:
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            last_role = None
            st.rerun()
