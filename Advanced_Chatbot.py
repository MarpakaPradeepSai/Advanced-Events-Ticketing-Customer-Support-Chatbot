import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time
import threading
from queue import Queue

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
    "{{APP}}": "<b>App</b>",
    "{{ASSISTANCE_SECTION}}": "<b>Assistance Section</b>",
    "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    # ... (keep all other static placeholders the same)
    "{{WEBSITE_URL}}": "www.events-ticketing.com"
}

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

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

class GenerationStopper:
    def __init__(self):
        self._stop = False
    
    def stop(self):
        self._stop = True
    
    def reset(self):
        self._stop = False
    
    @property
    def stopped(self):
        return self._stop

def generate_response(model, tokenizer, instruction, stopper, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    
    output_queue = Queue()
    
    def _generate():
        generated = []
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=[lambda *args, **kwargs: stopper.stopped]
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = response.find("Response:") + len("Response:")
        output_queue.put(response[response_start:].strip())
    
    thread = threading.Thread(target=_generate)
    thread.start()
    
    while thread.is_alive():
        time.sleep(0.1)
        if stopper.stopped:
            break
    
    thread.join()
    return output_queue.get() if not output_queue.empty() else "Generation stopped by user."

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

.horizontal-line {
    border-top: 2px solid #e0e0e0;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 43px;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "generation_stopper" not in st.session_state:
    st.session_state.generation_stopper = GenerationStopper()

example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?",
    "How can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?", 
    "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation?",
    "How can I sell my ticket?"
]

if not st.session_state.show_chat:
    st.markdown("""
    <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb;">
        <h1 style="font-size: 36px; color: #721c24; font-weight: bold; text-align: center;">⚠️Disclaimer</h1>
        <!-- Keep disclaimer content the same -->
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun()

if st.session_state.show_chat:
    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    selected_query = st.selectbox(
        "Choose a query from examples:",
        ["Choose your question"] + example_queries,
        key="query_selectbox",
        label_visibility="collapsed"
    )
    process_query_button = st.button("Ask this question", key="query_button")

    nlp = load_spacy_model()
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the model.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    last_role = None

    for message in st.session_state.chat_history:
        if message["role"] == "user" and last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        last_role = message["role"]

    if process_query_button:
        if selected_query == "Choose your question":
            st.error("⚠️ Please select your question from the dropdown.")
        elif selected_query:
            st.session_state.generation_stopper.reset()
            prompt = selected_query
            prompt = prompt[0].upper() + prompt[1:] if prompt else prompt

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
                    <button class="stop-button" onclick="window.parent.postMessage({'type': 'STOP_GENERATION'}, '*')">⏹️</button>
                </div>
                """
                generating_placeholder.markdown(generating_html, unsafe_allow_html=True)
                
                dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
                response = generate_response(model, tokenizer, prompt, st.session_state.generation_stopper)
                
                generating_placeholder.empty()
                full_response = replace_placeholders(response, dynamic_placeholders, static_placeholders)
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
            
            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            last_role = "assistant"

    if prompt := st.chat_input("Enter your own question:"):
        st.session_state.generation_stopper.reset()
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
                    <button class="stop-button" onclick="window.parent.postMessage({'type': 'STOP_GENERATION'}, '*')">⏹️</button>
                </div>
                """
                generating_placeholder.markdown(generating_html, unsafe_allow_html=True)
                
                dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
                response = generate_response(model, tokenizer, prompt, st.session_state.generation_stopper)
                
                generating_placeholder.empty()
                full_response = replace_placeholders(response, dynamic_placeholders, static_placeholders)
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
            
            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            last_role = "assistant"

    if st.session_state.chat_history:
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            st.session_state.generation_stopper.reset()
            last_role = None
            st.rerun()

components.html(
    """
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'STOP_GENERATION') {
            window.parent.postMessage({
                type: 'setComponentValue',
                componentId: 'stop_generation',
                value: true
            }, '*');
        }
    });
    </script>
    """,
    height=0
)

if 'stop_generation' in st.session_state and st.session_state.stop_generation:
    st.session_state.generation_stopper.stop()
    st.session_state.stop_generation = False
