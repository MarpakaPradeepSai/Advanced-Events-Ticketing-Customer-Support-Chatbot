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
            # Add progress indicator
            with st.spinner(f"Downloading {filename}..."):
                try:
                    response = requests.get(url, stream=True, timeout=30) # Added stream and timeout
                    response.raise_for_status() # Raise an exception for bad status codes
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024 # 1 Kibibyte
                    progress_bar = st.progress(0)
                    written = 0
                    with open(local_path, "wb") as f:
                        for data in response.iter_content(block_size):
                            written += len(data)
                            f.write(data)
                            if total_size > 0:
                                progress_bar.progress(min(written / total_size, 1.0))
                    progress_bar.empty() # Remove progress bar after completion
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to download {filename}: {e}")
                    # Clean up partially downloaded file
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    return False
        # else:
            # st.info(f"{filename} already exists.") # Optional: uncomment to see which files exist
    return True


# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_trf")
        return nlp
    except OSError:
        st.warning("Downloading spaCy model 'en_core_web_trf'. This may take a moment...")
        spacy.cli.download("en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")
        return nlp

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading AI model...") # Updated spinner message
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    st.info("Checking for model files...")
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot proceed without the model.")
        return None, None

    st.info("Model files downloaded/verified. Loading model and tokenizer...")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        st.success("AI Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        st.error("Please ensure all model files were downloaded correctly and are not corrupted.")
        return None, None


# Define static placeholders
static_placeholders = {
    "{{APP}}": "<b>App</b>",
    "{{ASSISTANCE_SECTION}}": "<b>Assistance Section</b>",
    "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_SECTION}}": "<b>Ticket Cancellation</b>",
    "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>",
    "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>",
    "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>",
    "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CANCELLATION_SECTION}}": "<b>Track Cancellation</b>",
    "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>",
    "{{CHECK_CANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>",
    "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>",
    "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>",
    "{{CONNECT_WITH_ORGANIZER}}": "<b>Connect with Organizer</b>",
    "{{CONTACT_SECTION}}": "<b>Contact</b>",
    "{{CONTACT_SUPPORT_LINK}}": "www.support-team.com",
    "{{CURRENT_TICKET_DETAILS}}": "<b>Current Ticket Details</b>",
    "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>",
    "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}": "<b>Customer Support</b>",
    "{{CUSTOMER_SUPPORT_SECTION}}": "<b>Customer Support</b>",
    "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{DELIVERY_SECTION}}": "<b>Delivery</b>",
    "{{EDIT_BUTTON}}": "<b>Edit</b>",
    "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",
    "{{EVENTS_PAGE}}": "<b>Events</b>",
    "{{EVENTS_SECTION}}": "<b>Events</b>",
    "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>",
    "{{GET_REFUND_BUTTON}}": "<b>Get Refund</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    "{{HELP_SECTION}}": "<b>Help</b>",
    "{{PAYMENT_ISSUE_OPTION}}": "<b>Payment Issue</b>",
    "{{PAYMENT_METHOD}}": "<b>Payment</b>",
    "{{PAYMENT_OPTION}}": "<b>Payment</b>",
    "{{PAYMENT_SECTION}}": "<b>Payments</b>",
    "{{PAYMENTS_HELP_SECTION}}": "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}": "<b>Payments</b>",
    "{{PRIVACY_POLICY_LINK}}": "<b>Privacy Policy</b>",
    "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>",
    "{{REFUND_SECTION}}": "<b>Refund</b>",
    "{{REFUND_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{REPORT_PAYMENT_PROBLEM}}": "<b>Report Payment</b>",
    "{{SAVE_BUTTON}}": "<b>Save</b>",
    "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>",
    "{{SEND_BUTTON}}": "<b>Send</b>",
    "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note: Typo here, space before SECTION
    "{{SUPPORT_CONTACT_LINK}}": "www.support-team.com",
    "{{SUPPORT_SECTION}}": "<b>Support</b>",
    "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{TICKET_AVAILABILITY_TAB}}": "<b>Ticket Availability</b>",
    "{{TICKET_DETAILS}}": "<b>Ticket Details</b>",
    "{{TICKET_INFORMATION}}": "<b>Ticket Information</b>",
    "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>",
    "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>",
    "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>",
    "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>",
    "{{TICKET_TRANSFER_TAB}}": "<b>Ticket Transfer</b>",
    "{{TICKETING_PAGE}}": "<b>Ticketing</b>",
    "{{TICKETS_TAB}}": "<b>Tickets</b>",
    "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>",
    "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>",
    "{{UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>",
    "{{UPGRADE_TICKET_INFORMATION}}": "<b>Ticket Upgradation</b>",
    "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>",
    "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>",
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
    # Prioritize finding EVENT and GPE specifically
    found_event = False
    found_city = False
    for ent in doc.ents:
        if not found_event and ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
            found_event = True
        elif not found_city and ent.label_ == "GPE": # GPE (Geopolitical Entity) often includes cities
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
            found_city = True
        elif not found_event and ent.label_ in ["WORK_OF_ART", "PRODUCT", "ORG"]: # Fallback for event-like names
             event_text = ent.text.title()
             dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
             found_event = True

    # If specific entities weren't found, use generic fallbacks
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "the event" # More natural fallback
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "your city" # More natural fallback
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Format consistent with fine-tuning (if applicable) or common practice
    input_text = f"Instruction: {instruction}\nResponse:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length).to(device) # Added truncation
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length, # Use max_new_tokens for clearer control over output length
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode only the newly generated tokens, excluding the prompt
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    # Simple stripping should be enough now
    return response.strip()

# CSS styling
st.markdown(
    """
<style>
/* General Button Styling */
.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71); /* Default gradient for most buttons */
    color: white !important;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 1.1em; /* Slightly adjusted size */
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

/* Specific style for the 'Ask this question' button */
/* Select the button within the horizontal block likely containing the selectbox and button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */
    font-size: 1.0em; /* Maybe slightly smaller than other buttons */
    padding: 8px 18px; /* Adjust padding if needed */
}

/* Specific style for the 'Reset Chat' button */
/* Target button by key if possible, otherwise use position or more specific selectors if needed */
/* This assumes it might be the last button on the page or use nth-of-type if structure is consistent */
/* A more robust way is to wrap it in a container with a unique class */
/* Let's try a general approach first, might need adjustment */
div.stButton:has(button[kind="secondary"]) > button, /* Streamlit > 1.3 Button Kind */
div.stButton:last-of-type > button { /* Fallback: Target last button, less reliable */
     /* background: linear-gradient(90deg, #6c757d, #343a40); /* Grey gradient */
     /* background: #dc3545; /* Simple red */
     /* Let's use Streamlit's default secondary look if possible, or customize */
     /* font-size: 0.9em; */ /* Smaller text */
     /* min-width: 80px; */
     /* If you want to force a specific look: */
      background: linear-gradient(90deg, #B0B0B0, #808080) !important; /* Grey gradient override */
      color: white !important;
      font-weight: normal !important;
}


/* Apply Times New Roman to all text elements */
* {
    font-family: 'Times New Roman', Times, serif !important;
}
/* Adjust specific elements if needed */
.stSelectbox *, .stTextInput *, .stTextArea *, .stChatMessage *, .stAlert * {
    font-family: 'Times New Roman', Times, serif !important;
}
.st-emotion-cache-r421ms { /* Example class for st.error, st.warning, etc. */
    font-family: 'Times New Roman', Times, serif !important;
}
.streamlit-expanderHeader *, .streamlit-expanderContent * {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Container for the Continue button to align it right */
.continue-button-container {
    text-align: right;
    margin-top: 20px; /* Add some space above the button */
    margin-bottom: 10px; /* Add some space below the button */
}
/* Style for the Continue button itself (can reuse general .stButton styles or add specific ones) */
.continue-button-container .stButton>button {
     background: linear-gradient(90deg, #28a745, #218838); /* Green gradient for continue */
     font-size: 1.2em; /* Make it prominent */
     padding: 12px 25px;
}


/* Horizontal line separator */
.horizontal-line {
    border-top: 1px solid #e0e0e0; /* Thinner line */
    margin: 10px 0; /* Adjust spacing */
}
</style>
    """,
    unsafe_allow_html=True,
)


# Streamlit UI
st.markdown("<h1 style='font-size: 43px;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state for controlling disclaimer visibility
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


# --- Model Loading ---
# Perform model loading before showing the disclaimer or chat
# Only load if it hasn't been loaded successfully before
if not st.session_state.model_loaded:
    nlp = load_spacy_model() # Load Spacy first (usually faster)
    model, tokenizer = load_model_and_tokenizer()
    if model is not None and tokenizer is not None and nlp is not None:
        st.session_state.model_loaded = True
        st.session_state.model = model # Store in session state if needed elsewhere
        st.session_state.tokenizer = tokenizer
        st.session_state.nlp = nlp
    else:
        st.error("Essential models could not be loaded. The chatbot cannot function.")
        st.stop() # Stop execution if models fail


# Example queries for dropdown
example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?",
    "How can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?",
    "What is the ticket cancellation fee?",
    "Can I sell my ticket?"
]

# Display Disclaimer and Continue button if chat hasn't started
if not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb;">
            <h2 style="font-size: 28px; color: #721c24; font-weight: bold; text-align: center;">⚠️ Disclaimer</h2>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents and may not be able to respond accurately to all types of queries.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                The chatbot is optimized to handle the following intents:
            </p>
            <ul style="font-size: 16px; line-height: 1.6; color: #721c24; margin-left: 20px;">
                <li>Cancel Ticket</li>
                <li>Buy Ticket</li>
                <li>Sell Ticket</li>
                <li>Transfer Ticket</li>
                <li>Upgrade Ticket</li>
                <li>Find Ticket</li>
                <li>Change Personal Details on Ticket</li>
                <li>Get Refund</li>
                <li>Find Upcoming Events</li>
                <li>Customer Service</li>
                <li>Check Cancellation Fee</li>
                <li>Track Cancellation</li>
                <li>Ticket Information</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                Please note that this chatbot may not be able to assist with queries outside of these predefined intents. Click "Continue" to start chatting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # --- Continue Button Section ---
    # Use the container div with the CSS class to align the button
    st.markdown('<div class="continue-button-container">', unsafe_allow_html=True)
    if st.button("Continue", key="continue_button"):
        st.session_state.show_chat = True
        st.rerun() # Rerun the script to hide disclaimer and show chat
    st.markdown('</div>', unsafe_allow_html=True)
    # --- End Continue Button Section ---

# Show chat interface only after clicking Continue
elif st.session_state.show_chat and st.session_state.model_loaded:
    # Retrieve models from session state
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    nlp = st.session_state.nlp

    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # --- Dropdown and Button section ---
    col1, col2 = st.columns([3, 1]) # Adjust ratio as needed
    with col1:
        selected_query = st.selectbox(
            "Choose a query from examples:",
            ["Choose your question"] + example_queries,
            index=0, # Default to "Choose your question"
            key="query_selectbox",
            label_visibility="collapsed" # Hide label for cleaner look
        )
    with col2:
        process_query_button = st.button("Ask this", key="query_button") # Shorter button text

    # Add a small space
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)


    last_role = None # Track last message role

    # Display chat messages from history
    chat_container = st.container() # Use a container to hold chat messages
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            is_last_message = (i == len(st.session_state.chat_history) - 1)
            # Add separator line only between user and assistant messages, not before the very first message
            if message["role"] == "user" and i > 0 and st.session_state.chat_history[i-1]["role"] == "assistant":
                 st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"], unsafe_allow_html=True)
            last_role = message["role"] # Keep track of the last displayed role


    # --- Handle Button Click ---
    if process_query_button:
        if selected_query != "Choose your question":
            prompt_to_process = selected_query
            # Add user message to history and display
            st.session_state.chat_history.append({"role": "user", "content": prompt_to_process, "avatar": "👤"})
            # Generate and add assistant response
            with st.spinner("Generating response..."):
                 dynamic_placeholders = extract_dynamic_placeholders(prompt_to_process, nlp)
                 response_gpt = generate_response(model, tokenizer, prompt_to_process)
                 full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            # Clear the selectbox choice after processing
            st.session_state.query_selectbox = "Choose your question"
            st.rerun() # Rerun to update the displayed chat
        else:
            st.toast("⚠️ Please select a question from the dropdown first.", icon="⚠️")


    # --- Handle Text Input ---
    if prompt := st.chat_input("Enter your own question:"):
        prompt_to_process = prompt[0].upper() + prompt[1:] if prompt else prompt # Capitalize first letter

        # Add user message to history and display
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_process, "avatar": "👤"})

        # Generate and add assistant response
        with st.spinner("Generating response..."):
             dynamic_placeholders = extract_dynamic_placeholders(prompt_to_process, nlp)
             response_gpt = generate_response(model, tokenizer, prompt_to_process)
             full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        st.rerun() # Rerun to update the displayed chat

    # --- Reset Button ---
    # Place reset button maybe in the sidebar or less prominently at the bottom
    st.sidebar.title("Options")
    if st.sidebar.button("Reset Chat", key="reset_button_sidebar"):
        st.session_state.chat_history = []
        st.session_state.query_selectbox = "Choose your question" # Reset selectbox too
        st.rerun()

    # Or at the bottom of the main area if preferred:
    # if st.session_state.chat_history:
    #     st.markdown("---") # Add a separator
    #     if st.button("Reset Chat", key="reset_button_main"):
    #         st.session_state.chat_history = []
    #         st.session_state.query_selectbox = "Choose your question" # Reset selectbox too
    #         st.rerun()


# Handle case where models failed to load but chat was attempted
elif not st.session_state.model_loaded:
     st.error("Chat cannot be displayed because the AI models failed to load.")
