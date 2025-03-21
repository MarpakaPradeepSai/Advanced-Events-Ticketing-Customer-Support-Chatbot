import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import spacy
import os
import requests
import warnings

warnings.filterwarnings('ignore')

# GitHub directory URL for model files
GITHUB_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model/"

# List of model files to download from GitHub
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json"
]

# Local directory to store downloaded model files
LOCAL_MODEL_DIR = "./DistilGPT2_Model"

# Static placeholders (shortened; add the rest as needed)
static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{CANCEL_TICKET_SECTION}}": "<b>Cancel Ticket</b>",
    # Add the rest here
}

# Function to download model files from GitHub with streaming
def download_model_files():
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR)
    
    for file_name in MODEL_FILES:
        file_url = GITHUB_URL + file_name
        local_path = os.path.join(LOCAL_MODEL_DIR, file_name)
        if not os.path.exists(local_path):
            try:
                response = requests.get(file_url, stream=True)
                if response.status_code == 200:
                    response.raw.decode_content = True
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    st.error(f"Failed to download {file_name} from GitHub. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"Error downloading {file_name}: {str(e)}")

# Load the GPT-2 model and tokenizer with caching
@st.cache_resource
def load_gpt2_model():
    download_model_files()
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_DIR)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

# Load the SpaCy model for NER with caching
@st.cache_resource
def load_spacy_model():
    import spacy.cli
    spacy.cli.download("en_core_web_trf")
    nlp = spacy.load("en_core_web_trf")
    return nlp

# Extract dynamic placeholders using SpaCy NER
def extract_dynamic_placeholders(instruction, nlp):
    doc = nlp(instruction)
    dynamic_placeholders = {}
    
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            dynamic_placeholders['{{EVENT}}'] = f"<b>{ent.text.title()}</b>"
        elif ent.label_ == "GPE":
            dynamic_placeholders['{{CITY}}'] = f"<b>{ent.text.title()}</b>"
    
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"
    
    return dynamic_placeholders

# Replace placeholders in the response
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Generate response using the fine-tuned model
def generate_response(instruction, tokenizer, model, nlp, max_length=256):
    model.eval()
    dynamic_placeholders = extract_dynamic_placeholders(instruction, nlp)
    
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    raw_response = response[response_start:].strip()
    
    final_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)
    return final_response

# Streamlit UI
def main():
    st.title("Advanced Events Ticketing Chatbot")
    st.write("Ask me anything about event ticketing!")

    # Load models
    tokenizer, model = load_gpt2_model()
    nlp = load_spacy_model()

    # User input
    user_input = st.text_input("Enter your question:", "")
    
    if st.button("Submit"):
        if user_input:
            user_input = user_input[0].upper() + user_input[1:]
            with st.spinner("Generating response..."):
                response = generate_response(user_input, tokenizer, model, nlp)
            st.write("**Chatbot Response:**")
            st.markdown(response, unsafe_allow_html=True)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
