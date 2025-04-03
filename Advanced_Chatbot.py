import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList
import requests
import os
import spacy
import time
import uuid # For unique keys

# --- Constants and Setup (Keep as before) ---
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

# --- Function Definitions (Keep download_model_files, load_spacy_model, load_model_and_tokenizer, static_placeholders, replace_placeholders, extract_dynamic_placeholders) ---

# Function to download model files from GitHub
def download_model_files(model_dir="/tmp/DistilGPT2_Model"):
    os.makedirs(model_dir, exist_ok=True)
    progress_bar = st.progress(0, text="Downloading model files...")
    total_files = len(MODEL_FILES)
    downloaded_count = 0

    for i, filename in enumerate(MODEL_FILES):
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

        if not os.path.exists(local_path):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() # Raise an exception for bad status codes

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 # 1 Kibibyte
                current_size = 0

                with open(local_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        f.write(data)
                        current_size += len(data)
                        # Update overall progress based on file count
                        file_progress = (i + (current_size / total_size if total_size > 0 else 1)) / total_files
                        progress_text = f"Downloading {filename} ({current_size/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)..." if total_size > 0 else f"Downloading {filename}..."
                        progress_bar.progress(min(file_progress, 1.0), text=progress_text)

                downloaded_count += 1
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename}: {e}")
                # Clean up partially downloaded file if error occurs
                if os.path.exists(local_path):
                    os.remove(local_path)
                progress_bar.empty() # Remove progress bar on error
                return False
        else:
            downloaded_count += 1
            # Update progress even if file exists
            progress_bar.progress((i + 1) / total_files, text=f"Checked {filename} (already exists)...")

    if downloaded_count == total_files:
        progress_bar.progress(1.0, text="Model files download/check complete!")
        time.sleep(1) # Keep message visible briefly
        progress_bar.empty() # Remove progress bar on success
        return True
    else:
        progress_bar.empty() # Remove progress bar if failed somehow (should be caught earlier)
        return False


# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_trf")
        return nlp
    except OSError:
        st.warning("SpaCy 'en_core_web_trf' model not found. Downloading...")
        spacy.cli.download("en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")
        st.success("SpaCy model downloaded successfully.")
        return nlp

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading AI Model...") # More informative spinner
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    # Download files first, show progress
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot load model.")
        return None, None

    # Now load from the local directory
    try:
        # Load model with trust_remote_code=True if necessary, but be cautious
        # If the config/model files are standard GPT2/DistilGPT2, it might not be needed
        # Let's try without it first for better security, enable if loading fails specifically due to it
        model = GPT2LMHeadModel.from_pretrained(model_dir) # trust_remote_code=True may be needed depending on exact source/config
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        st.error("If the error mentions 'trust_remote_code', you might need to uncomment that argument in the 'from_pretrained' calls, but understand the security implications.")
        # Example of enabling it if needed:
        # model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        # tokenizer = GPT2Tokenizer.from_pretrained(model_dir, trust_remote_code=True)
        return None, None


# Define static placeholders (keep as before)
static_placeholders = {
    "{{APP}}": "<b>App</b>", "{{ASSISTANCE_SECTION}}": "<b>Assistance Section</b>", "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>", "{{CANCEL_TICKET_SECTION}}": "<b>Ticket Cancellation</b>",
    "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>", "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>",
    "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>", "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CANCELLATION_SECTION}}": "<b>Track Cancellation</b>", "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>",
    "{{CHECK_CANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>", "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>",
    "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>", "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>",
    "{{CONNECT_WITH_ORGANIZER}}": "<b>Connect with Organizer</b>", "{{CONTACT_SECTION}}": "<b>Contact</b>",
    "{{CONTACT_SUPPORT_LINK}}": "<a href='#' target='_blank'>www.support-team.com</a>", # Make links clickable
    "{{CURRENT_TICKET_DETAILS}}": "<b>Current Ticket Details</b>", "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>",
    "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>", "{{CUSTOMER_SUPPORT_PORTAL}}": "<b>Customer Support</b>",
    "{{CUSTOMER_SUPPORT_SECTION}}": "<b>Customer Support</b>", "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{DELIVERY_SECTION}}": "<b>Delivery</b>", "{{EDIT_BUTTON}}": "<b>Edit</b>", "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",
    "{{EVENTS_PAGE}}": "<b>Events</b>", "{{EVENTS_SECTION}}": "<b>Events</b>", "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>", "{{GET_REFUND_BUTTON}}": "<b>Get Refund</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>", "{{HELP_SECTION}}": "<b>Help</b>", "{{PAYMENT_ISSUE_OPTION}}": "<b>Payment Issue</b>",
    "{{PAYMENT_METHOD}}": "<b>Payment</b>", "{{PAYMENT_OPTION}}": "<b>Payment</b>", "{{PAYMENT_SECTION}}": "<b>Payments</b>",
    "{{PAYMENTS_HELP_SECTION}}": "<b>Payments Help</b>", "{{PAYMENTS_PAGE}}": "<b>Payments</b>",
    "{{PRIVACY_POLICY_LINK}}": "<b>Privacy Policy</b>", "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>", "{{REFUND_SECTION}}": "<b>Refund</b>", "{{REFUND_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{REPORT_PAYMENT_PROBLEM}}": "<b>Report Payment</b>", "{{SAVE_BUTTON}}": "<b>Save</b>", "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>", "{{SEND_BUTTON}}": "<b>Send</b>", "{{SUPPORT_ SECTION}}": "<b>Support</b>",
    "{{SUPPORT_CONTACT_LINK}}": "<a href='#' target='_blank'>www.support-team.com</a>", # Make links clickable
    "{{SUPPORT_SECTION}}": "<b>Support</b>",
    "{{SUPPORT_TEAM_LINK}}": "<a href='#' target='_blank'>www.support-team.com</a>", # Make links clickable
    "{{TICKET_AVAILABILITY_TAB}}": "<b>Ticket Availability</b>", "{{TICKET_DETAILS}}": "<b>Ticket Details</b>",
    "{{TICKET_INFORMATION}}": "<b>Ticket Information</b>", "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>",
    "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>", "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>", "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>",
    "{{TICKET_TRANSFER_TAB}}": "<b>Ticket Transfer</b>", "{{TICKETING_PAGE}}": "<b>Ticketing</b>",
    "{{TICKETS_TAB}}": "<b>Tickets</b>", "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>", "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>",
    "{{UPGRADE_OPTION}}": "<b>Upgrade</b>", "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>",
    "{{UPGRADE_TICKET_INFORMATION}}": "<b>Ticket Upgradation</b>", "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>", "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>",
    "{{WEBSITE_URL}}": "<a href='#' target='_blank'>www.events-ticketing.com</a>" # Make links clickable
}


# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    # Replace dynamic first to avoid conflicts if dynamic value contains a static key
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    doc = nlp(user_question)
    dynamic_placeholders = {}
    # Use more specific labels if available in your fine-tuned NER or training data
    # Sticking to broader categories for this example
    event_found = False
    location_found = False
    for ent in doc.ents:
        # Prioritize ORG for potential event names if EVENT is not reliable
        if ent.label_ in ["EVENT", "ORG"] and not event_found:
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
            event_found = True
        # Use GPE (Geopolitical Entity) for cities/locations
        elif ent.label_ == "GPE" and not location_found:
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
            location_found = True
        # Add more entity types if relevant (DATE, TIME, PERSON etc.)
        # elif ent.label_ == "DATE":
        #     dynamic_placeholders['{{DATE}}'] = f"<b>{ent.text}</b>"

    # Fallbacks if entities are not found
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "the event" # More generic fallback
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "your city" # More generic fallback
    return dynamic_placeholders


# --- Stopping Criteria for Generation ---
class StopGenerationCriteria(StoppingCriteria):
    def __init__(self, stop_flag_key: str):
        super().__init__()
        self.stop_flag_key = stop_flag_key

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check the session state flag
        if st.session_state.get(self.stop_flag_key, False):
            # print(f"Stop signal received via {self.stop_flag_key}!") # Debugging
            # Optionally reset the flag here, or do it after generation call
            # st.session_state[self.stop_flag_key] = False
            return True  # Signal to stop generation
        return False # Signal to continue

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256, stopping_criteria=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Make sure input_text is well-formed for the fine-tuned model
    # Adjust this prefix based on how the model was trained (e.g., Alpaca format, etc.)
    # Assuming a simple instruction/response format based on the original code
    input_text = f"Instruction: {instruction}\nResponse:" # Added newline for clarity

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device) # Added truncation

    # Ensure stopping_criteria is a list if provided
    stopping_criteria_list = StoppingCriteriaList([stopping_criteria]) if stopping_criteria else None

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length, # Use max_new_tokens for better control
            # max_length=inputs["input_ids"].shape[1] + max_length, # Alternative way using max_length
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria_list # Pass the criteria here
        )

    # Decode the response, excluding the prompt
    # Find the start of the response based on the input length
    input_length = inputs["input_ids"].shape[1]
    # Decode only the newly generated tokens
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)

    # Clean up potential artifacts or the prompt remnants if decoding wasn't perfect
    # The model might sometimes repeat parts of the prompt or add unwanted prefixes
    response_marker = "Response:"
    res_start_index = response.find(response_marker)
    if res_start_index != -1:
         response = response[res_start_index + len(response_marker):].strip()
    else:
         # If marker not found (might happen with stopping criteria), just strip
         response = response.strip()

    return response


# --- CSS Styling (Keep as before) ---
st.markdown(
    """
<style>
/* General Button Styling */
.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71); /* Stylish gradient */
    color: white !important; /* Ensure text is white */
    border: none;
    border-radius: 25px; /* Rounded corners */
    padding: 10px 20px; /* Padding */
    font-size: 1.1em; /* Adjusted Font size */
    font-weight: bold; /* Bold text */
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease; /* Smooth transitions */
    display: inline-flex; /* Helps with alignment */
    align-items: center;
    justify-content: center;
    margin-top: 5px; /* Adjust slightly if needed */
    width: auto; /* Fit content width */
    min-width: 100px; /* Optional: ensure a minimum width */
    font-family: 'Times New Roman', Times, serif !important; /* Times New Roman */
}
.stButton>button:hover {
    transform: scale(1.05); /* Slightly larger on hover */
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); /* Shadow on hover */
    color: white !important; /* Ensure text stays white on hover */
}
.stButton>button:active {
    transform: scale(0.98); /* Slightly smaller when clicked */
}

/* Specific style for 'Ask this question' button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */
    color: white !important;
    font-size: 1em; /* Slightly smaller */
    padding: 8px 18px;
}

/* Style for the 'Stop Generation' Button (placed dynamically) */
/* We'll target it via a container or specific key later if needed, */
/* but for now, let's assume it might inherit .stButton */
/* Or give it a specific class if we wrap it */
.stop-button button {
    background: #d9534f !important; /* Red color for stop */
    color: white !important;
    border-radius: 50% !important; /* Circle */
    width: 35px !important; /* Fixed size */
    height: 35px !important; /* Fixed size */
    min-width: 35px !important;
    padding: 0 !important; /* Remove padding */
    font-size: 1.2em !important; /* Adjust icon size */
    line-height: 35px !important; /* Center icon vertically */
    text-align: center !important;
    margin-left: 10px; /* Space from spinner text */
    border: 1px solid #d43f3a !important;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2);
}
.stop-button button:hover {
    background: #c9302c !important;
    border-color: #ac2925 !important;
    transform: scale(1.1);
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
}


/* Apply Times New Roman to all text elements */
body, * {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Adjustments for specific Streamlit elements */
.stSelectbox div[data-baseweb="select"] > div {
    font-family: 'Times New Roman', Times, serif !important;
}
.stTextInput input {
    font-family: 'Times New Roman', Times, serif !important;
}
.stTextArea textarea {
    font-family: 'Times New Roman', Times, serif !important;
}
.stChatMessage {
    font-family: 'Times New Roman', Times, serif !important;
}
.stMarkdown {
     font-family: 'Times New Roman', Times, serif !important;
}
/* Ensure Markdown within chat also uses the font */
.stChatMessage .stMarkdown p, .stChatMessage .stMarkdown li, .stChatMessage .stMarkdown b, .stChatMessage .stMarkdown a {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Horizontal Line */
.horizontal-line {
    border-top: 1px solid #e0e0e0; /* Thinner line */
    margin: 10px 0; /* Adjust spacing */
}

/* Chat Input Shadow */
div[data-testid="stChatInput"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Softer shadow */
    border-radius: 8px;
    padding: 10px 15px;
    margin: 15px 0;
    background-color: #f9f9f9; /* Slightly off-white background */
}

/* Styling for generating message container */
.generating-container {
    display: flex;
    align-items: center;
    justify-content: space-between; /* Push button to the right */
    width: 100%;
    background-color: #f0f2f6; /* Light background for contrast */
    padding: 8px 12px;
    border-radius: 5px;
    margin-bottom: 5px; /* Space below */
}
.generating-container .stSpinner {
    margin-right: 10px; /* Space between spinner text and potential button */
}
</style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit UI ---
st.markdown("<h1 style='font-size: 43px;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "stop_generation_flag" not in st.session_state:
    st.session_state.stop_generation_flag = False # Global flag for stopping
if "generating" not in st.session_state:
    st.session_state.generating = False # Track if generation is in progress
if "current_stop_key" not in st.session_state:
    st.session_state.current_stop_key = None # Track which stop flag key is active

# Example queries
example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the upcoming Music Festival in London?", # Added example event/city
    "How do I change my personal details on my ticket?",
    "How can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?",
    "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation?",
    "How can I sell my ticket?"
]

# --- Disclaimer Logic (Keep as before) ---
if not st.session_state.show_chat:
    # Disclaimer content... (kept concise for brevity)
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb; font-family: 'Times New Roman', Times, serif !important;">
            <h1 style="font-size: 36px; color: #721c24; font-weight: bold; text-align: center;">⚠️ Disclaimer</h1>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                This <b>Chatbot</b> is designed for ticketing inquiries but has limitations. It's optimized for intents like:
            </p>
            <ul style="font-size: 14px; line-height: 1.5; color: #721c24; padding-left: 20px;">
                <li>Buy/Sell/Cancel/Transfer/Upgrade Ticket</li>
                <li>Find Ticket/Upcoming Events</li>
                <li>Change Personal Details</li>
                <li>Get Refund/Check Cancellation Fee/Track Cancellation</li>
                <li>Customer Service/Ticket Information</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                Responses outside these areas may be less accurate. Please be patient if the bot struggles.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun()

# --- Chat Interface Logic ---
if st.session_state.show_chat:

    # Load models only when chat starts
    with st.spinner("Initializing chatbot components..."):
        nlp = load_spacy_model()
        model, tokenizer = load_model_and_tokenizer()

    if model is None or tokenizer is None or nlp is None:
        st.error("Failed to initialize chatbot components. Please check logs or try refreshing.")
        st.stop()

    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # --- Dropdown and Button Section ---
    col1, col2 = st.columns([4, 1]) # Adjust ratio if needed
    with col1:
         selected_query = st.selectbox(
             "Choose a query from examples:",
             ["Choose your question"] + example_queries,
             key="query_selectbox",
             label_visibility="collapsed"
         )
    with col2:
        process_query_button = st.button("Ask this", key="query_button", help="Ask the selected example question")


    # --- Chat History Display ---
    last_role = None
    for i, message in enumerate(st.session_state.chat_history):
        # Add separator line between user and assistant turns
        if message["role"] == "user" and last_role == "assistant":
             st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

        with st.chat_message(message["role"], avatar=message.get("avatar")):
             # Check if this is an assistant message that might be in progress
             # Note: This logic is tricky with Streamlit reruns. The "generating" state
             # mainly helps control the *new* message generation, not re-rendering old ones.
             # For simplicity, just display the final content stored.
             st.markdown(message["content"], unsafe_allow_html=True)

        last_role = message["role"]

    # --- Function to handle message processing ---
    def process_and_display_message(user_input):
        global last_role # Use global last_role tracker

        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input, "avatar": "👤"})
        if last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input, unsafe_allow_html=True)
        last_role = "user"

        # Prepare for assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            stop_key = f"stop_flag_{uuid.uuid4()}" # Unique key for this specific generation
            st.session_state[stop_key] = False # Initialize stop flag for this key
            st.session_state.current_stop_key = stop_key # Track the active key
            st.session_state.generating = True # Mark generation as started

            # --- Container for Spinner and Stop Button ---
            gen_container = st.container()
            with gen_container:
                gen_cols = st.columns([10, 1]) # Columns for spinner text and button
                with gen_cols[0]:
                    spinner_placeholder = st.empty()
                    spinner_placeholder.markdown("Generating response...") # Use markdown for potential styling
                with gen_cols[1]:
                    stop_button_placeholder = st.empty()
                    # Add the stop button *inside* the column
                    # The button click will set the state and cause a rerun
                    if stop_button_placeholder.button("⏹️", key=f"stop_btn_{stop_key}", help="Stop Generation", type="secondary"):
                        st.session_state[stop_key] = True
                        st.session_state.generating = False # Mark as no longer generating (or trying to stop)
                        st.session_state.current_stop_key = None
                        # print(f"Stop button pressed for key {stop_key}") # Debug
                        spinner_placeholder.markdown("Stopping...")
                        stop_button_placeholder.empty() # Remove button after click
                        # We don't rerun here; the StoppingCriteria will handle it during the generate call check
                        # Or if generate() finished before the click, the flag prevents issues on next run.

            # --- Generation Logic ---
            try:
                # Prepare stopping criteria
                stop_criteria = StopGenerationCriteria(stop_flag_key=stop_key)

                # Run generation (can take time)
                dynamic_placeholders = extract_dynamic_placeholders(user_input, nlp)
                response_gpt = generate_response(model, tokenizer, user_input, stopping_criteria=stop_criteria)

                # Check if stopped
                was_stopped = st.session_state.get(stop_key, False)

                full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                if was_stopped and not full_response.strip(): # Handle case where stop happened very early
                    full_response = "[Generation stopped by user]"
                elif was_stopped:
                    full_response += "\n\n*[Generation stopped by user]*" # Append indication

            except Exception as e:
                st.error(f"Error during response generation: {e}")
                full_response = "Sorry, I encountered an error while generating the response."
                was_stopped = False # Ensure we don't show stop message on error

            finally:
                # --- Cleanup and Display ---
                st.session_state.generating = False # Mark generation as finished/stopped
                st.session_state.current_stop_key = None
                if stop_key in st.session_state: # Clean up the specific flag
                     del st.session_state[stop_key]

                gen_container.empty() # Remove spinner/button container
                message_placeholder.markdown(full_response, unsafe_allow_html=True) # Display final result

                # Add final response to history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
                last_role = "assistant"

    # --- Handle Dropdown Query Submission ---
    if process_query_button:
        if selected_query == "Choose your question":
            st.toast("⚠️ Please select a question from the dropdown.", icon="💡")
        elif selected_query:
            # Prevent generation if already generating
            if st.session_state.generating and st.session_state.current_stop_key:
                 st.toast("Please wait for the current response or stop it first.", icon="⏳")
            else:
                prompt_from_dropdown = selected_query
                # Capitalize first letter
                prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown
                process_and_display_message(prompt_from_dropdown)
                # Clear dropdown selection after processing (optional)
                # st.session_state.query_selectbox = "Choose your question" # This might cause immediate rerun, test carefully
                st.rerun() # Rerun to update the chat display fully

    # --- Handle User Input from Chat Box ---
    if prompt := st.chat_input("Enter your own question:"):
        # Capitalize first letter
        prompt = prompt[0].upper() + prompt[1:] if prompt else prompt

        if not prompt.strip():
            st.toast("⚠️ Please enter a question.", icon="✍️")
        else:
            # Prevent generation if already generating
            if st.session_state.generating and st.session_state.current_stop_key:
                 st.toast("Please wait for the current response or stop it first.", icon="⏳")
            else:
                process_and_display_message(prompt)
                st.rerun() # Rerun to update the chat display fully

    # --- Reset Chat Button ---
    if st.session_state.chat_history:
         # Add some space before the reset button
         st.markdown("<br>", unsafe_allow_html=True)
         if st.button("Reset Chat", key="reset_button"):
             st.session_state.chat_history = []
             st.session_state.generating = False # Reset generation state
             st.session_state.current_stop_key = None
             # Clear any lingering stop flags (optional, but good practice)
             keys_to_delete = [k for k in st.session_state if k.startswith("stop_flag_")]
             for k in keys_to_delete:
                 del st.session_state[k]
             last_role = None
             st.rerun()
