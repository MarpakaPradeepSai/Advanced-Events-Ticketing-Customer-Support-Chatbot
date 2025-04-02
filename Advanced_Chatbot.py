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

# Function to download model files from GitHub
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

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Check your internet connection or GitHub URL.")
        return None, None

    model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Define static placeholders
static_placeholders = {
    "{{APP}}": "<b>App</b>",
    # ... (keep all static placeholders identical)
    "{{WEBSITE_URL}}": "www.events-ticketing.com"
}

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    doc = nlp(user_question)
    dynamic_placeholders = {}
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE":
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
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

# CSS styling with shadow effect for input section
st.markdown(
    """
<style>
.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71);
    color: white !important;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 1.2em;
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: 5px;
    width: auto;
    min-width: 100px;
    font-family: 'Times New Roman', Times, serif !important;
}

/* Added shadow effect for input section */
[data-testid="stChatInputContainer"] {
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
    background-color: white;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
    color: white !important;
}

.stButton>button:active {
    transform: scale(0.98);
}

* {
    font-family: 'Times New Roman', Times, serif !important;
}

.stSelectbox > div > div > div > div {
    font-family: 'Times New Roman', Times, serif !important;
}

.stTextInput > div > div > input {
    font-family: 'Times New Roman', Times, serif !important;
}

.stTextArea > div > div > textarea {
    font-family: 'Times New Roman', Times, serif !important;
}

.stChatMessage {
    font-family: 'Times New Roman', Times, serif !important;
}

.st-emotion-cache-r421ms {
    font-family: 'Times New Roman', Times, serif !important;
}

.streamlit-expanderContent {
    font-family: 'Times New Roman', Times, serif !important;
}

div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6);
    color: white !important;
}

.horizontal-line {
    border-top: 2px solid #e0e0e0;
    margin: 15px 0;
}
</style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI (remainder identical)
st.markdown("<h1 style='font-size: 43px;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    # ... (keep example queries identical)
    "Can I sell my ticket?"
]

if not st.session_state.show_chat:
    # ... (keep disclaimer section identical)
    if st.button("Continue", key="continue_button"):
        st.session_state.show_chat = True
        st.rerun()

if st.session_state.show_chat:
    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # ... (keep dropdown and button section identical)

    nlp = load_spacy_model()
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        st.error("Failed to load the model.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    last_role = None

    # ... (keep chat history display identical)

    if process_query_button:
        # ... (keep query processing identical)

    # Input box with shadow effect
    if prompt := st.chat_input("Enter your own question:"):
        # ... (keep input processing identical)

    if st.session_state.chat_history:
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            last_role = None
            st.rerun()
