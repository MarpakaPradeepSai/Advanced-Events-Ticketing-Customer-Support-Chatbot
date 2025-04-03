import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time
import threading # Needed for potential future interruption (though not fully implemented here)

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

# --- Model Loading and Utility Functions (Mostly Unchanged) ---

# Function to download model files from GitHub
def download_model_files(model_dir="/tmp/DistilGPT2_Model"):
    os.makedirs(model_dir, exist_ok=True)
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(MODEL_FILES)

    for i, filename in enumerate(MODEL_FILES):
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)
        status_text.text(f"Downloading {filename}...")

        if not os.path.exists(local_path):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 # 1 Kibibyte
                downloaded_size = 0
                with open(local_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        downloaded_size += len(data)
                        f.write(data)
                        # Update progress within file download if needed (optional, can slow down)
                        # if total_size > 0:
                        #     file_progress = downloaded_size / total_size
                        #     overall_progress = (i + file_progress) / total_files
                        #     progress_bar.progress(min(overall_progress, 1.0))

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from GitHub. Error: {e}")
                # Clean up partially downloaded file
                if os.path.exists(local_path):
                    os.remove(local_path)
                status_text.text("")
                progress_bar.empty()
                return False
        progress_bar.progress((i + 1) / total_files)

    status_text.text("Model files downloaded successfully.")
    time.sleep(1) # Keep success message visible briefly
    status_text.empty()
    progress_bar.empty()
    return True

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        st.info("Downloading spaCy model 'en_core_web_trf'...")
        spacy.cli.download("en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")
    return nlp

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading AI Model...") # More informative spinner
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot proceed.")
        st.stop() # Stop execution if download fails

    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        st.error("Please check if the downloaded files are complete and valid.")
        # Optionally, try deleting the cached directory and asking user to retry
        # import shutil
        # if os.path.exists(model_dir):
        #     shutil.rmtree(model_dir)
        # st.warning("Attempting to clear cached model directory. Please refresh the page to retry download.")
        st.stop()


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
    "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note: Space before SECTION here
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
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE": # GPE (Geopolitical Entity) often captures cities/locations
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
        # Add more entity types if needed (e.g., DATE, ORG)
    # Provide defaults if not found
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "the event" # More generic default
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "your city" # More generic default
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
# *** IMPORTANT NOTE ***: True interruption of model.generate() is complex and
# often requires modifying the underlying library or using lower-level process control.
# This implementation *simulates* stopping by checking a flag *before* and *after*
# the generation call. The computation might still finish in the background
# if the stop button is clicked mid-generation.
def generate_response(model, tokenizer, instruction, max_length=256):
    # Check if stop was requested *before* starting generation
    if st.session_state.get("stop_requested", False):
        return None # Indicate generation was stopped

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

    try:
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
                # eos_token_id=tokenizer.eos_token_id # Ensure generation stops at EOS
            )
        # Check if stop was requested *during* generation (less effective for blocking calls)
        if st.session_state.get("stop_requested", False):
             return None # Indicate generation was stopped

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Find the response part more reliably
        response_marker = "Response:"
        response_start_index = response.find(response_marker)
        if response_start_index != -1:
            final_response = response[response_start_index + len(response_marker):].strip()
        else:
            # Fallback if "Response:" marker is not found (e.g., model generates something unexpected)
            # Try to remove the instruction part if possible
            if response.startswith(input_text.replace(" Response:", "")):
                 final_response = response[len(input_text.replace(" Response:", "")):].strip()
            else:
                 final_response = response # Return the whole output as a fallback

        return final_response

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, I encountered an error while generating the response."


# --- CSS Styling (Unchanged) ---
st.markdown(
    """
<style>
.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71); /* Stylish gradient */
    color: white !important; /* Ensure text is white */
    border: none;
    border-radius: 25px; /* Rounded corners */
    padding: 10px 20px; /* Padding */
    font-size: 1.2em; /* Font size */
    font-weight: bold; /* Bold text */
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease; /* Smooth transitions */
    display: inline-flex; /* Helps with alignment */
    align-items: center;
    justify-content: center;
    margin-top: 5px; /* Adjust slightly if needed for alignment with selectbox */
    width: auto; /* Fit content width */
    min-width: 100px; /* Optional: ensure a minimum width */
    font-family: 'Times New Roman', Times, serif !important; /* Times New Roman for buttons */
}
.stButton>button:hover {
    transform: scale(1.05); /* Slightly larger on hover */
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); /* Shadow on hover */
    color: white !important; /* Ensure text stays white on hover */
}
.stButton>button:active {
    transform: scale(0.98); /* Slightly smaller when clicked */
}

/* Apply Times New Roman to all text elements */
* {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Specific adjustments for Streamlit elements if needed */
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
.st-emotion-cache-r421ms { /* Example class for st.error, st.warning, etc. */
    font-family: 'Times New Roman', Times, serif !important;
}
.streamlit-expanderContent { /* For text inside expanders */
    font-family: 'Times New Roman', Times, serif !important;
}

/* Style for the "Ask this question" button specifically */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */
    color: white !important;
}

/* Horizontal line separator */
    .horizontal-line {
        border-top: 2px solid #e0e0e0;
        margin: 15px 0;
    }

/* Chat Input Shadow Effect */
div[data-testid="stChatInput"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0; /* Adjusted margin */
    position: relative; /* Needed for potential absolute positioning of stop button */
}

/* Style for the Stop button container */
.stop-button-container {
    text-align: center; /* Center the button */
    margin-top: -5px; /* Pull it up slightly closer to input */
    margin-bottom: 10px;
    height: 40px; /* Reserve space even when hidden */
}

/* Style for the Stop button itself */
.stop-button-container .stButton>button {
    background-color: #dc3545 !important; /* Red background */
    background-image: none !important; /* Override gradient */
    border: 1px solid #dc3545;
    color: white !important;
    padding: 5px 15px; /* Smaller padding */
    font-size: 1em; /* Smaller font size */
    min-width: auto; /* Allow button to be smaller */
    border-radius: 15px; /* Slightly less rounded */
    font-weight: normal; /* Normal weight */
}
.stop-button-container .stButton>button:hover {
    background-color: #c82333 !important; /* Darker red on hover */
    border: 1px solid #bd2130;
    box-shadow: 0px 3px 10px rgba(0, 0, 0, 0.2);
    transform: scale(1.03);
}
.stop-button-container .stButton>button:active {
    background-color: #bd2130 !important;
    transform: scale(0.99);
}

</style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit UI ---
st.markdown("<h1 style='font-size: 43px;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# --- Session State Initialization ---
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False # Tracks if the model is currently generating
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False # Flag to signal stopping generation

# --- Load Models ---
# Moved model loading here to happen after potential disclaimer
# Only load if chat is intended to be shown or already shown
if st.session_state.show_chat or 'model' not in globals(): # Load only if needed
    nlp = load_spacy_model()
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        # Error handled within load_model_and_tokenizer using st.stop()
        pass

# --- Disclaimer Logic ---
if not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb; font-family: Arial, sans-serif;">
            <h1 style="font-size: 36px; color: #721c24; font-weight: bold; text-align: center;">⚠️Disclaimer</h1>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents, and may not be able to respond accurately to all types of queries.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                The chatbot is optimized to handle the following intents:
            </p>
            <ul style="font-size: 16px; line-height: 1.6; color: #721c24;">
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
                Please note that this chatbot may not be able to assist with queries outside of these predefined intents.
                Even if the model fails to provide accurate responses from the predefined intents, we kindly ask for your patience and encourage you to try again.
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

# --- Main Chat Interface Logic ---
elif st.session_state.show_chat:
    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # Example queries section
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
    col1, col2 = st.columns([3, 1]) # Adjust column ratio if needed
    with col1:
        selected_query = st.selectbox(
            "Choose a query from examples:",
            ["Choose your question"] + example_queries,
            key="query_selectbox",
            label_visibility="collapsed"
        )
    with col2:
        process_query_button = st.button("Ask this question", key="query_button", disabled=st.session_state.is_generating)

    # Display chat history
    last_role = None
    for message in st.session_state.chat_history:
        is_user = message["role"] == "user"
        # Add separator only between user and assistant messages
        if is_user and last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        last_role = message["role"]

    # Container for the Stop button - placed between history and input
    stop_button_placeholder = st.empty()
    with stop_button_placeholder.container():
        if st.session_state.is_generating:
            st.markdown('<div class="stop-button-container">', unsafe_allow_html=True)
            if st.button("⏹️ Stop Generation", key="stop_button"):
                st.session_state.stop_requested = True
                st.session_state.is_generating = False # Immediately reflect stop
                # No rerun here, allow current script execution to potentially finish
                # The check within generate_response and after will handle it.
                st.toast("Stop request sent. Finishing current step...", icon="🛑")
                # We might need a rerun later if generation is truly async, but not now.
                # st.rerun() # Use cautiously, might interrupt display updates
            st.markdown('</div>', unsafe_allow_html=True)
        # else:
            # Keep the container to maintain layout consistency, but empty
            # st.markdown('<div class="stop-button-container"></div>', unsafe_allow_html=True)


    # --- Function to handle sending a message and getting response ---
    def handle_message(prompt_text):
        if not prompt_text:
            st.toast("⚠️ Please enter a question.")
            return

        prompt_text = prompt_text[0].upper() + prompt_text[1:] # Capitalize

        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": prompt_text, "avatar": "👤"})
        # Need to rerun to display user message immediately before generation starts
        st.session_state.is_generating = True
        st.session_state.stop_requested = False # Reset stop flag for new request
        st.rerun() # Rerun to show user message and stop button

    # --- Logic for processing after rerun when is_generating is True ---
    if st.session_state.is_generating and not st.session_state.stop_requested:
        # Find the last user message to generate response for
        last_user_message = None
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break

        if last_user_message:
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                generating_response_text = "Generating response..."
                full_response = "..." # Default in case of error

                try:
                    with st.spinner(generating_response_text):
                        # Check stop flag again just before the call
                        if st.session_state.stop_requested:
                             full_response = "Generation stopped by user."
                        else:
                            dynamic_placeholders = extract_dynamic_placeholders(last_user_message, nlp)
                            response_gpt = generate_response(model, tokenizer, last_user_message)

                            # Check stop flag immediately after the call
                            if st.session_state.stop_requested or response_gpt is None:
                                full_response = "Generation stopped by user."
                            else:
                                full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                                if not full_response: # Handle empty generation
                                     full_response = "Sorry, I couldn't generate a specific response for that. Can you please rephrase?"

                    message_placeholder.markdown(full_response, unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    full_response = "Sorry, there was an error processing your request."
                    message_placeholder.markdown(full_response, unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
                finally:
                    # Generation finished or was stopped, reset flags
                    st.session_state.is_generating = False
                    st.session_state.stop_requested = False
                    # Rerun one last time to update the UI state (hide stop button, etc.)
                    st.rerun() # This ensures the stop button disappears correctly

    # --- Input Handling ---

    # Handle example query button press
    if process_query_button and selected_query != "Choose your question":
        handle_message(selected_query)
    elif process_query_button and selected_query == "Choose your question":
         st.toast("⚠️ Please select your question from the dropdown.")


    # Handle chat input
    if prompt := st.chat_input("Enter your own question:", key="chat_input_box", disabled=st.session_state.is_generating):
        handle_message(prompt)


    # --- Reset Button ---
    if st.session_state.chat_history:
        if st.button("Reset Chat", key="reset_button", disabled=st.session_state.is_generating):
            st.session_state.chat_history = []
            st.session_state.is_generating = False
            st.session_state.stop_requested = False
            st.rerun()
