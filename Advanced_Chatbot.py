# streamlit run your_script_name.py
import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# GitHub directory containing the DistilGPT2 model files
GITHUB_BASE_URL = "https://raw.githubusercontent.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/main/DistilGPT2_Model"
MODEL_DIR = "/tmp/DistilGPT2_Model"

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

# Define static placeholders
static_placeholders = {
    "{{APP}}": "<b>App</b>", "{{ASSISTANCE_SECTION}}": "<b>Assistance Section</b>", "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>", "{{CANCEL_TICKET_SECTION}}": "<b>Ticket Cancellation</b>", "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>",
    "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>", "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>", "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CANCELLATION_SECTION}}": "<b>Track Cancellation</b>", "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>", "{{CHECK_CANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>", "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>", "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>",
    "{{CONNECT_WITH_ORGANIZER}}": "<b>Connect with Organizer</b>", "{{CONTACT_SECTION}}": "<b>Contact</b>", "{{CONTACT_SUPPORT_LINK}}": "www.support-team.com",
    "{{CURRENT_TICKET_DETAILS}}": "<b>Current Ticket Details</b>", "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>", "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}": "<b>Customer Support</b>", "{{CUSTOMER_SUPPORT_SECTION}}": "<b>Customer Support</b>", "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{DELIVERY_SECTION}}": "<b>Delivery</b>", "{{EDIT_BUTTON}}": "<b>Edit</b>", "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",
    "{{EVENTS_PAGE}}": "<b>Events</b>", "{{EVENTS_SECTION}}": "<b>Events</b>", "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>", "{{GET_REFUND_BUTTON}}": "<b>Get Refund</b>", "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    "{{HELP_SECTION}}": "<b>Help</b>", "{{PAYMENT_ISSUE_OPTION}}": "<b>Payment Issue</b>", "{{PAYMENT_METHOD}}": "<b>Payment</b>",
    "{{PAYMENT_OPTION}}": "<b>Payment</b>", "{{PAYMENT_SECTION}}": "<b>Payments</b>", "{{PAYMENTS_HELP_SECTION}}": "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}": "<b>Payments</b>", "{{PRIVACY_POLICY_LINK}}": "<b>Privacy Policy</b>", "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>", "{{REFUND_SECTION}}": "<b>Refund</b>", "{{REFUND_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{REPORT_PAYMENT_PROBLEM}}": "<b>Report Payment</b>", "{{SAVE_BUTTON}}": "<b>Save</b>", "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>", "{{SEND_BUTTON}}": "<b>Send</b>", "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note: space in original key
    "{{SUPPORT_CONTACT_LINK}}": "www.support-team.com", "{{SUPPORT_SECTION}}": "<b>Support</b>", "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{TICKET_AVAILABILITY_TAB}}": "<b>Ticket Availability</b>", "{{TICKET_DETAILS}}": "<b>Ticket Details</b>", "{{TICKET_INFORMATION}}": "<b>Ticket Information</b>",
    "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>", "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>", "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>", "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>", "{{TICKET_TRANSFER_TAB}}": "<b>Ticket Transfer</b>",
    "{{TICKETING_PAGE}}": "<b>Ticketing</b>", "{{TICKETS_TAB}}": "<b>Tickets</b>", "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>", "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>", "{{UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>", "{{UPGRADE_TICKET_INFORMATION}}": "<b>Ticket Upgradation</b>", "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>", "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>", "{{WEBSITE_URL}}": "www.events-ticketing.com"
}

# --- Utility Functions ---

# Function to download model files from GitHub
def download_model_files(model_dir=MODEL_DIR):
    """Downloads model files if they don't exist locally."""
    logger.info(f"Checking model files in {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    all_files_present = True

    for filename in MODEL_FILES:
        url = f"{GITHUB_BASE_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

        if not os.path.exists(local_path):
            all_files_present = False
            logger.info(f"Downloading {filename} from {url}...")
            try:
                response = requests.get(url, stream=True, timeout=60) # Added stream and timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Successfully downloaded {filename}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from GitHub. Error: {e}")
                # Clean up partially downloaded file if exists
                if os.path.exists(local_path):
                    os.remove(local_path)
                return False # Stop download process on failure
        # else: logger.debug(f"File {filename} already exists.") # Optional: uncomment for debugging

    if all_files_present:
        logger.info("All model files are present locally.")
    return True

# Load spaCy model for NER
@st.cache_resource(show_spinner="Loading NLP model...")
def load_spacy_model():
    """Loads the spaCy model, downloading if necessary."""
    model_name = "en_core_web_trf"
    try:
        logger.info(f"Loading spaCy model: {model_name}")
        # Check if model is installed, if not, download it
        if not spacy.util.is_package(model_name):
            logger.info(f"spaCy model {model_name} not found. Downloading...")
            spacy.cli.download(model_name)
            logger.info(f"Finished downloading {model_name}.")
        nlp = spacy.load(model_name)
        logger.info("spaCy model loaded successfully.")
        return nlp
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        st.error(f"Could not load spaCy NLP model ({model_name}). Please check your internet connection and try again. Error details: {e}")
        return None

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading Chatbot model...")
def load_model_and_tokenizer(model_dir=MODEL_DIR):
    """Loads the Transformers model and tokenizer after downloading."""
    logger.info("Attempting to load model and tokenizer...")
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot load the chatbot model.")
        return None, None

    try:
        logger.info(f"Loading model from {model_dir}...")
        # trust_remote_code=True might be needed depending on the model source/config
        # If the model is standard DistilGPT2 fine-tuned, it might not be necessary.
        # Keep it if it was required for your specific fine-tuned version.
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        logger.info("Model loaded successfully.")

        logger.info(f"Loading tokenizer from {model_dir}...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        # Ensure pad token is set for tokenizer if not already present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id # Also update model config
            logger.info("Set tokenizer pad_token to eos_token.")

        logger.info("Tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}", exc_info=True)
        st.error(f"Failed to load the chatbot model from {model_dir}. Error: {e}")
        return None, None

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders_map):
    """Replaces static and dynamic placeholders in the response text."""
    for placeholder, value in static_placeholders_map.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    """Extracts EVENT and GPE entities as dynamic placeholders."""
    if nlp is None:
        return {"{{EVENT}}": "event", "{{CITY}}": "city"} # Default if NLP model failed

    dynamic_placeholders = {}
    try:
        doc = nlp(user_question)
        for ent in doc.ents:
            # Using title case for better presentation
            if ent.label_ == "EVENT":
                event_text = ent.text.strip().title()
                dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                logger.info(f"Extracted EVENT: {event_text}")
            elif ent.label_ == "GPE": # Geographical Entity (cities, countries)
                city_text = ent.text.strip().title()
                dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
                logger.info(f"Extracted GPE (City): {city_text}")

        # Provide default values if not found
        if '{{EVENT}}' not in dynamic_placeholders:
            dynamic_placeholders['{{EVENT}}'] = "event"
        if '{{CITY}}' not in dynamic_placeholders:
            dynamic_placeholders['{{CITY}}'] = "city"

    except Exception as e:
        logger.error(f"Error during NER processing: {e}")
        # Fallback to defaults in case of error
        dynamic_placeholders['{{EVENT}}'] = "event"
        dynamic_placeholders['{{CITY}}'] = "city"

    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    """Generates a response using the loaded GPT-2 model."""
    if model is None or tokenizer is None:
        return "Error: Chatbot model is not loaded."

    try:
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        input_text = f"Instruction: {instruction} Response:"
        logger.info(f"Generating response for input: {input_text}")

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length // 2) # Truncate input if too long
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generation parameters
        gen_kwargs = {
            "max_length": max_length,
            "num_return_sequences": 1,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }

        # Generate response
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, **gen_kwargs)

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the response part after "Response:"
        response_start_idx = response.find("Response:")
        if response_start_idx != -1:
            clean_response = response[response_start_idx + len("Response:"):].strip()
        else:
            # Fallback if "Response:" marker is not found (e.g., model didn't follow format)
            # Try to remove the input instruction part
            if response.startswith(input_text.replace(" Response:", "")): # Check if output includes input
                 clean_response = response[len(input_text.replace(" Response:", "")):].strip()
            else:
                 clean_response = response.strip() # Use the whole output as a last resort


        logger.info(f"Generated response: {clean_response}")
        return clean_response

    except Exception as e:
        logger.error(f"Error during response generation: {e}", exc_info=True)
        return f"Sorry, I encountered an error while generating the response: {e}"

# --- CSS Styling ---
st.markdown(
    """
<style>
/* General Button Styling */
.stButton>button {
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 1.1em; /* Adjusted size */
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
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2); /* Softer shadow */
}
.stButton>button:active {
    transform: scale(0.98);
}

/* Specific Button: Ask this question (using nth-of-type might be fragile, consider keys or classes if possible) */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Blue gradient */
    color: white !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
     color: white !important;
}


/* Specific Button: Reset Chat */
div[data-testid="stBottom"] div[data-testid="stButton"] button { /* Target button in bottom block */
    background: linear-gradient(90deg, #ff8a00, #e52e71); /* Original orange/pink gradient */
    color: white !important;
    margin-top: 15px; /* Add space above reset button */
}
div[data-testid="stBottom"] div[data-testid="stButton"] button:hover {
    color: white !important;
}

/* Specific Button: Continue */
/* Let's assume the continue button is the first button directly under the columns */
div[data-testid="stVerticalBlock"] > div[style*="column-gap"] > div:nth-child(2) .stButton > button {
     background: linear-gradient(90deg, #4CAF50, #8BC34A); /* Green gradient */
     color: white !important;
     float: right; /* Align to right within its column */
}
div[data-testid="stVerticalBlock"] > div[style*="column-gap"] > div:nth-child(2) .stButton > button:hover {
     color: white !important;
}


/* Global Font */
* {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Input elements font */
.stSelectbox [data-baseweb="select"] > div,
.stTextInput input,
.stTextArea textarea {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Chat Messages Font */
.stChatMessage {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Chat Input Box Shadow */
div[data-testid="stChatInput"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Softer shadow */
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    border: 1px solid #e0e0e0; /* Subtle border */
}

/* Horizontal Line Separator */
.horizontal-line {
    border-top: 1px solid #eee; /* Lighter line */
    margin: 15px 0;
}

/* Disclaimer Box */
.disclaimer-box {
    background-color: #fff8e1; /* Light yellow */
    padding: 20px;
    border-radius: 10px;
    color: #6d4c41; /* Brown text */
    border: 1px solid #ffecb3; /* Light yellow border */
    font-family: 'Times New Roman', Times, serif !important; /* Ensure font */
    margin-bottom: 20px; /* Space below disclaimer */
}
.disclaimer-box h1 {
    font-size: 28px; /* Smaller header */
    color: #5d4037; /* Darker brown */
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
}
.disclaimer-box p, .disclaimer-box ul {
    font-size: 15px; /* Slightly smaller text */
    line-height: 1.6;
    color: #6d4c41;
    font-family: 'Times New Roman', Times, serif !important;
}
.disclaimer-box ul {
    list-style-type: disc;
    margin-left: 20px;
}
.disclaimer-box b {
    font-weight: bold;
    color: #5d4037; /* Darker bold text */
}

/* Generating Response Area */
.generating-container {
    display: flex;
    align-items: center;
    justify-content: space-between; /* Pushes items to ends */
    padding: 5px 10px;
    background-color: #f0f2f6; /* Light background */
    border-radius: 5px;
    margin-top: 5px;
    font-size: 0.9em;
    color: #555;
}
.generating-text {
    font-style: italic;
}
/* Stop Button (using emoji now, so less complex styling needed) */
.stop-button-container .stButton>button {
    background: #f0f2f6; /* Match container background */
    color: #dc3545 !important; /* Red color for stop */
    border: 1px solid #dc3545;
    border-radius: 5px;
    padding: 2px 8px; /* Smaller padding */
    font-size: 0.9em;
    min-width: auto; /* Allow button to be small */
    margin: 0; /* Remove default margin */
    box-shadow: none;
}
.stop-button-container .stButton>button:hover {
    background: #e9ecef;
    color: #dc3545 !important;
    transform: none; /* Disable hover scale */
    box-shadow: none;
}
.stop-button-container .stButton>button:active {
    transform: scale(0.95); /* Slight shrink on click */
}

</style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit App UI ---

st.markdown("<h1 style='font-size: 38px; text-align: center; color: #333;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# --- Session State Initialization ---
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_role" not in st.session_state:
    st.session_state.last_role = None # Track last message role
if "current_prompt_key" not in st.session_state:
    st.session_state.current_prompt_key = None # Track which input triggered generation

# --- Example Queries ---
example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the upcoming Music Festival in London?", # Added dynamic example
    "How do I change my personal details on my ticket?",
    "How can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?",
    "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation?",
    "How can I sell my ticket?"
]

# --- Disclaimer Screen ---
if not st.session_state.show_chat:
    st.markdown(
        """
        <div class="disclaimer-box">
            <h1>⚠️ Disclaimer</h1>
            <p>This <b>Chatbot</b> assists with ticketing inquiries but is fine-tuned on specific intents due to computational limits. It may not respond accurately to all queries.</p>
            <p>Optimized Intents:</p>
            <ul>
                <li>Cancel/Buy/Sell/Transfer/Upgrade/Find Ticket</li>
                <li>Change Personal Details</li>
                <li>Get Refund / Check Cancellation Fee / Track Cancellation</li>
                <li>Find Upcoming Events</li>
                <li>Customer Service / Ticket Information</li>
            </ul>
            <p>Queries outside these areas might not be handled correctly. We appreciate your patience if the chatbot struggles even with supported intents.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Continue button aligned to the right
    _, col2 = st.columns([4, 1]) # Adjust ratio if needed
    with col2:
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun() # Rerun to hide disclaimer and show chat

# --- Main Chat Interface ---
if st.session_state.show_chat:

    # Load Models (only once after clicking Continue)
    nlp = load_spacy_model()
    model, tokenizer = load_model_and_tokenizer()

    # Check if models loaded successfully before proceeding
    if nlp is None or model is None or tokenizer is None:
        st.error("Critical components failed to load. The chatbot cannot function. Please check logs or try refreshing.")
        st.stop() # Stop execution if models aren't loaded

    st.info("Ask me about ticket cancellations, refunds, or event inquiries!")

    # --- Example Query Selection (Top) ---
    col1, col2 = st.columns([3, 1]) # Ratio for selectbox and button
    with col1:
        selected_query = st.selectbox(
            "Or choose an example question:",
            [""] + example_queries, # Add empty option
            index=0, # Default to empty
            key="query_selectbox",
            label_visibility="collapsed" # Hide the label "Or choose..."
        )
    with col2:
        process_query_button = st.button("Ask this example", key="query_button")

    st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True) # Separator

    # --- Chat History Display ---
    chat_container = st.container() # Container to hold chat messages
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            # Add separator only between user and assistant turns
            if i > 0 and message["role"] == "user" and st.session_state.chat_history[i-1]["role"] == "assistant":
                 st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"], unsafe_allow_html=True)


    # --- Response Generation Logic ---
    def handle_generation(prompt, prompt_source_key):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        st.session_state.last_role = "user"
        st.session_state.current_prompt_key = prompt_source_key # Track source

        # Display user message immediately
        with chat_container: # Add to the main chat display
            if len(st.session_state.chat_history) > 1 and st.session_state.chat_history[-2]["role"] == "assistant":
                 st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt, unsafe_allow_html=True)

        # Prepare for assistant response
        with chat_container: # Add to the main chat display
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty() # For the final response
                generating_indicator_placeholder = st.empty() # For "Generating..." + Stop button

                # Set flags for generation state
                st.session_state.is_generating = True
                st.session_state.stop_generation = False # Ensure stop is reset

                # Callback function for the stop button
                def trigger_stop():
                    st.session_state.stop_generation = True
                    logger.info("Stop button clicked, setting stop_generation flag.")
                    # The rerun triggered by the button click will handle the rest

                # Display "Generating..." and Stop Button
                with generating_indicator_placeholder.container():
                    g_col1, g_col2 = st.columns([0.85, 0.15])
                    with g_col1:
                        g_col1.markdown('<div class="generating-text">Generating response...</div>', unsafe_allow_html=True)
                    with g_col2:
                        # Use a unique key based on the prompt source to avoid conflicts
                        stop_button_key = f"stop_{prompt_source_key}"
                        g_col2.button("Stop", key=stop_button_key, on_click=trigger_stop, help="Stop generation")
                        # Apply specific styling class if needed via markdown wrapper (though simpler button better)
                        #g_col2.markdown('<div class="stop-button-container">', unsafe_allow_html=True)
                        #g_col2.button("⏹️", key=stop_button_key, on_click=trigger_stop, help="Stop generation")
                        #g_col2.markdown('</div>', unsafe_allow_html=True)


        # === IMPORTANT: RERUN IS NEEDED AFTER THIS POINT ===
        # The button click itself causes a rerun. The logic below runs *after* the rerun.

    def complete_generation_or_stop(prompt):
        # This part runs *after* the potential stop button click and rerun
        if st.session_state.is_generating:
            # Retrieve placeholders where content will be placed
            # This needs careful handling as placeholders might be lost on rerun if not managed properly.
            # A simpler approach might be to just append the final/stopped message directly without placeholders.
            # Let's try appending directly:

            final_response_content = ""
            if st.session_state.stop_generation:
                logger.info("Generation was stopped.")
                final_response_content = "*Generation stopped by user.*"
                st.session_state.stop_generation = False # Reset flag
            else:
                logger.info("Proceeding with response generation.")
                try:
                    # Generate actual response
                    dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
                    response_gpt = generate_response(model, tokenizer, prompt)
                    final_response_content = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                except Exception as e:
                    logger.error(f"Exception during final generation/placeholder step: {e}", exc_info=True)
                    final_response_content = "Sorry, an error occurred while finalizing the response."

            # Append the final/stopped message to history
            st.session_state.chat_history.append({"role": "assistant", "content": final_response_content, "avatar": "🤖"})
            st.session_state.last_role = "assistant"
            st.session_state.is_generating = False
            st.session_state.current_prompt_key = None # Clear tracker

            # Rerun one last time to display the final/stopped message from history
            st.rerun()


    # --- Input Processing ---

    # Process example query button
    if process_query_button and selected_query:
        prompt = selected_query[0].upper() + selected_query[1:] # Capitalize
        logger.info(f"Example query button clicked for: {prompt}")
        handle_generation(prompt, "example_query")
        # Execution stops here, handle_generation prepares UI, next run completes it

    # Process chat input
    if user_input := st.chat_input("Enter your question here:", key="user_input_box"):
        prompt = user_input[0].upper() + user_input[1:] # Capitalize
        logger.info(f"User submitted input: {prompt}")
        handle_generation(prompt, "user_input")
        # Execution stops here, handle_generation prepares UI, next run completes it


    # --- Handle Completion/Stopping AFTER Rerun ---
    if st.session_state.is_generating and st.session_state.current_prompt_key:
        # Find the original prompt that started this generation
        # The last user message should correspond to the prompt
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            original_prompt = st.session_state.chat_history[-1]["content"]
            logger.info(f"Completing/Stopping generation for prompt: {original_prompt}")
            complete_generation_or_stop(original_prompt)
        else:
            # Fallback/Error state - should not happen if logic is correct
            logger.warning("In generating state but couldn't find last user prompt.")
            st.session_state.is_generating = False # Reset state
            st.session_state.current_prompt_key = None


    # --- Reset Chat Button (Bottom) ---
    if st.session_state.chat_history:
         # Add some space before the reset button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset Chat", key="reset_button"):
            logger.info("Resetting chat history.")
            st.session_state.chat_history = []
            st.session_state.last_role = None
            st.session_state.is_generating = False
            st.session_state.stop_generation = False
            st.session_state.current_prompt_key = None
            # Clear the selected query dropdown too
            st.session_state.query_selectbox = ""
            st.rerun() # Rerun to clear the chat display
