import streamlit as st
import torch
import spacy
import os
import requests
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ---------------------------- File Download Functions ----------------------------
def download_from_github(repo_url, file_name, save_path):
    file_url = f"{repo_url}/{file_name}"
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download {file_name} from GitHub. Status code: {response.status_code}")

# ---------------------------- Model Configuration ----------------------------
GITHUB_MODEL_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json"
]
model_dir = "./distilgpt2_model"

# Create model directory if not exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Download model files
for file in MODEL_FILES:
    download_from_github(GITHUB_MODEL_URL, file, os.path.join(model_dir, file))

# ---------------------------- NLP Models Loading ----------------------------
@st.cache_resource
def load_models():
    # Load spaCy NER model
    nlp = spacy.load("en_core_web_trf")
    
    # Load DistilGPT2 model and tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        model.eval()
        return nlp, model, tokenizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

nlp, model, tokenizer = load_models()
if model is None or tokenizer is None:
    st.stop()

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------- Placeholder Configuration ----------------------------
static_placeholders = {
    "{{WEBSITE_URL}}": "https://www.eventicketofficial.com",
    "{{CANCEL_TICKET_SECTION}}": "Ticket Management",
    "{{APP}}": "Eventicket Mobile App",
    "{{CONTACT_EMAIL}}": "support@eventicket.com"
}

# ---------------------------- Helper Functions ----------------------------
def extract_dynamic_placeholders(text):
    """Extract entities from text using spaCy NER"""
    doc = nlp(text)
    placeholders = {}
    
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            placeholders["{{EVENT}}"] = f"<b>{ent.text.title()}</b>"
        elif ent.label_ == "GPE":
            placeholders["{{CITY}}"] = f"<b>{ent.text.title()}</b>"
    
    # Set defaults if not found
    if "{{EVENT}}" not in placeholders:
        placeholders["{{EVENT}}"] = "<b>event</b>"
    if "{{CITY}}" not in placeholders:
        placeholders["{{CITY}}"] = "<b>city</b>"
    
    return placeholders

def replace_placeholders(text, dynamic_placeholders):
    """Replace both static and dynamic placeholders in text"""
    # Replace static placeholders first
    for placeholder, value in static_placeholders.items():
        text = text.replace(placeholder, f"<b>{value}</b>")
    
    # Replace dynamic placeholders
    for placeholder, value in dynamic_placeholders.items():
        text = text.replace(placeholder, value)
    
    return text

def generate_response(prompt):
    """Generate response using DistilGPT2"""
    input_text = f"Instruction: {prompt} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# ---------------------------- Streamlit UI ----------------------------
st.title("🎟️ Advanced Events Ticketing Chatbot")
st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Enter your question:"):
    # Format user input
    prompt = prompt[0].upper() + prompt[1:] if prompt else prompt
    
    if not prompt.strip():
        st.error("Please enter a valid question")
        st.stop()
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "avatar": "👤"
    })
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt, unsafe_allow_html=True)
    
    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Show thinking animation
        thinking_dots = ""
        for _ in range(3):
            thinking_dots += "."
            message_placeholder.markdown(f"Thinking{thinking_dots}")
            time.sleep(0.3)
        
        # Generate initial response
        raw_response = generate_response(prompt)
        
        # Process response
        dynamic_placeholders = extract_dynamic_placeholders(prompt)
        processed_response = replace_placeholders(raw_response, dynamic_placeholders)
        
        # Display final response
        message_placeholder.empty()
        st.markdown(processed_response, unsafe_allow_html=True)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": processed_response,
            "avatar": "🤖"
        })

# Reset chat button
if st.session_state.chat_history:
    st.markdown("""
    <style>
    .stButton>button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 1.2em;
        font-weight: bold;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()
