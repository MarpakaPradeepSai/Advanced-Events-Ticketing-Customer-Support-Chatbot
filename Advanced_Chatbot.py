import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time

# --- Constants and Setup ---

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

# Define static placeholders (truncated for brevity, assume full list is here)
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
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>", "{{SEND_BUTTON}}": "<b>Send</b>", "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note: potential typo here ("SUPPORT_ SECTION")
    "{{SUPPORT_CONTACT_LINK}}": "www.support-team.com", "{{SUPPORT_SECTION}}": "<b>Support</b>", "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{TICKET_AVAILABILITY_TAB}}": "<b>Ticket Availability</b>", "{{TICKET_DETAILS}}": "<b>Ticket Details</b>", "{{TICKET_INFORMATION}}": "<b>Ticket Information</b>",
    "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>", "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>", "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>", "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>", "{{TICKET_TRANSFER_TAB}}": "<b>Ticket Transfer</b>",
    "{{TICKETING_PAGE}}": "<b>Ticketing</b>", "{{TICKETS_TAB}}": "<b>Tickets</b>", "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>", "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>", "{{UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>", "{{UPGRADE_TICKET_INFORMATION}}": "<b>Ticket Upgradation</b>", "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>", "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>", "{{WEBSITE_URL}}": "www.events-ticketing.com"
}


# --- Functions ---

# Function to download model files from GitHub
def download_model_files(model_dir="/tmp/DistilGPT2_Model"):
    """Downloads model files if they don't exist locally."""
    os.makedirs(model_dir, exist_ok=True)
    all_files_exist = True
    for filename in MODEL_FILES:
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(local_path):
            all_files_exist = False
            st.info(f"Downloading {filename}...")
            try:
                url = f"{GITHUB_MODEL_URL}/{filename}"
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()  # Raise an exception for bad status codes
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success(f"Downloaded {filename}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename}: {e}")
                # Clean up potentially incomplete file
                if os.path.exists(local_path):
                    os.remove(local_path)
                return False # Stop download process if one file fails
    if not all_files_exist:
        st.success("All model files checked/downloaded.")
    return True

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model, attempting download if needed."""
    model_name = "en_core_web_trf"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        st.warning(f"SpaCy model '{model_name}' not found. Downloading...")
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
            st.success(f"SpaCy model '{model_name}' downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download spaCy model '{model_name}'. NER features will be limited. Error: {e}")
            return None
    return nlp

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading language model...")
def load_model_and_tokenizer():
    """Downloads (if needed) and loads the DistilGPT2 model and tokenizer."""
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot load model.")
        return None, None
    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        st.error("Please ensure all model files were downloaded correctly and are not corrupted.")
        return None, None

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    """Replaces static and dynamic placeholders in the response string."""
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    """Extracts EVENT and GPE entities as dynamic placeholders."""
    dynamic_placeholders = {}
    if nlp and user_question: # Check if nlp model loaded and question exists
        try:
            doc = nlp(user_question)
            for ent in doc.ents:
                if ent.label_ == "EVENT":
                    event_text = ent.text.title()
                    dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                elif ent.label_ == "GPE": # GPE typically refers to geopolitical entities like cities, countries
                    city_text = ent.text.title()
                    dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
        except Exception as e:
            st.warning(f"NER processing failed: {e}. Using default placeholders.")

    # Provide defaults if not found
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event" # Use a generic default
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city" # Use a generic default
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    """Generates a response using the loaded GPT-2 model."""
    if not model or not tokenizer:
        return "Error: Model not loaded."
    try:
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_text = f"Instruction: {instruction} Response:"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device) # Added truncation

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length, # Use max_new_tokens for better control
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Find the response part more robustly
        response_marker = "Response:"
        response_start_index = response.find(response_marker)
        if response_start_index != -1:
            return response[response_start_index + len(response_marker):].strip()
        else:
             # Fallback if "Response:" marker isn't found (e.g., if input_text was part of output)
             input_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
             return response[input_len:].strip() # Return text generated after input

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, I encountered an error while generating the response."

# --- CSS Styling ---
st.markdown(
    """
<style>
/* Button Styles */
.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71); /* Stylish gradient */
    color: white !important; /* Ensure text is white */
    border: none;
    border-radius: 25px; /* Rounded corners */
    padding: 10px 20px; /* Padding */
    font-size: 1em; /* Adjusted size */
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

/* Specific style for "Ask this question" button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button[kind="secondary"] { /* More specific selector */
    background: linear-gradient(90deg, #29ABE2, #0077B6) !important; /* Different gradient */
    color: white !important;
}

/* Apply Times New Roman to common text elements */
body, .stApp, .stMarkdown, .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    font-family: 'Times New Roman', Times, serif !important;
}
/* Ensure chat messages use the font */
.stChatMessage {
    font-family: 'Times New Roman', Times, serif !important;
}
.stChatMessage * { /* Apply to elements within chat messages */
    font-family: inherit !important;
}
/* Style error/warning/info boxes */
.stAlert {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Horizontal line separator */
.horizontal-line {
    border-top: 2px solid #e0e0e0;
    margin: 15px 0;
}

/* Chat Input Shadow Effect */
div[data-testid="stChatInput"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); /* Softer shadow */
    border-radius: 8px; /* Slightly more rounded */
    padding: 8px 12px; /* Adjust padding */
    margin: 10px 0;
    background-color: #f9f9f9; /* Light background */
    border: 1px solid #eee; /* Subtle border */
}

/* Custom styling for selectbox and its button */
div[data-testid="stHorizontalBlock"] {
    align-items: end; /* Align items to the bottom */
}

div[data-testid="stHorizontalBlock"] .stSelectbox {
    margin-right: 10px; /* Add space between selectbox and button */
}

</style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit App ---

st.markdown("<h1 style='font-size: 43px; font-family: \"Times New Roman\", Times, serif;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# --- Initialization ---
# Initialize session state variables
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_query" not in st.session_state:
    st.session_state.processing_query = None # Stores the query to be processed

# --- Disclaimer ---
if not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb; font-family: 'Times New Roman', Times, serif;">
            <h2 style="font-size: 28px; color: #721c24; font-weight: bold; text-align: center;">⚠️ Disclaimer</h2>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                This <b>Chatbot</b> is designed to assist with ticketing inquiries but is fine-tuned on specific intents due to computational limits. It may not accurately respond to all query types.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                Optimized intents include: Cancel/Buy/Sell/Transfer/Upgrade/Find Ticket, Change Details, Get Refund, Find Events, Customer Service, Check Cancellation Fee/Track Cancellation, Ticket Information.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                Queries outside these areas may not be handled well. Your patience is appreciated if the chatbot struggles, even with supported intents.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Continue button aligned to the right
    col1_disc, col2_disc = st.columns([4, 1])
    with col2_disc:
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun() # Rerun to hide disclaimer and show chat

# --- Main Chat Interface ---
if st.session_state.show_chat:

    # Load Models (runs only once thanks to cache)
    nlp = load_spacy_model()
    model, tokenizer = load_model_and_tokenizer()

    if model is None or tokenizer is None:
        st.error("Chatbot cannot function without the language model. Please check error messages above and ensure model files are accessible.")
        st.stop() # Stop execution if model loading failed

    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # --- Example Queries Section ---
    example_queries = [
        "How do I buy a ticket?", "How can I upgrade my ticket for the upcoming event in Hyderabad?",
        "How do I change my personal details on my ticket?", "How can I find details about upcoming events?",
        "How do I contact customer service?", "How do I get a refund?", "What is the ticket cancellation fee?",
        "How can I track my ticket cancellation?", "How can I sell my ticket?"
    ]

    # Use columns for better layout of selectbox and button
    col1_ex, col2_ex = st.columns([3, 1]) # Adjust ratio as needed

    with col1_ex:
        selected_query = st.selectbox(
            "Choose a query from examples:",
            ["Choose your question"] + example_queries,
            key="query_selectbox",
            label_visibility="collapsed"
        )

    with col2_ex:
        # Use a unique key for this button
        process_query_button = st.button("Ask this question", key="ask_example_button", type="secondary") # Use type secondary for potential different styling


    # --- Process Button Click (Set state, don't process here) ---
    if process_query_button:
        if selected_query != "Choose your question":
            # Set the query to be processed in the next run, if none is already set
            if st.session_state.processing_query is None:
                 # Capitalize first letter
                capitalized_query = selected_query[0].upper() + selected_query[1:] if selected_query else selected_query
                st.session_state.processing_query = capitalized_query
                 # No rerun here, let Streamlit's natural flow handle it
            else:
                 st.toast("Please wait for the current response to complete.", icon="⏳")
        else:
            st.toast("⚠️ Please select a question from the dropdown.", icon="⚠️")

    # --- Chat Input (Set state, don't process here) ---
    if prompt := st.chat_input("Enter your own question:", key="chat_input_box"):
        prompt_strip = prompt.strip()
        if prompt_strip:
             # Set the query to be processed in the next run, if none is already set
             if st.session_state.processing_query is None:
                 # Capitalize first letter
                 capitalized_prompt = prompt_strip[0].upper() + prompt_strip[1:] if prompt_strip else prompt_strip
                 st.session_state.processing_query = capitalized_prompt
                 # No rerun here
             else:
                 st.toast("Please wait for the current response to complete.", icon="⏳")
        else:
            st.toast("⚠️ Please enter a question.", icon="⚠️")


    # --- Display Chat History ---
    last_role_displayed = None # Track last message role *during display loop*
    for i, message in enumerate(st.session_state.chat_history):
        # Add separator line if user message follows an assistant message
        if message["role"] == "user" and last_role_displayed == "assistant" and i > 0:
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

        # Display the message
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        last_role_displayed = message["role"]


    # --- Process Stored Query (The Core Logic) ---
    # This block runs *after* history is displayed and *after* inputs might have set processing_query
    if st.session_state.processing_query:
        query_to_process = st.session_state.processing_query
        # Clear the flag *before* processing so button/input works again after completion
        st.session_state.processing_query = None

        # Add User message to history and display it immediately
        if not st.session_state.chat_history or st.session_state.chat_history[-1].get("content") != query_to_process:
             # Add separator if needed
             if st.session_state.chat_history and st.session_state.chat_history[-1].get("role") == "assistant":
                 st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

             st.session_state.chat_history.append({"role": "user", "content": query_to_process, "avatar": "👤"})
             with st.chat_message("user", avatar="👤"):
                 st.markdown(query_to_process, unsafe_allow_html=True)
             last_role_displayed = "user" # Update display tracker

        # Generate and display Assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = "Thinking..." # Initial text
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

            with st.spinner("Generating response..."):
                # Perform NER
                dynamic_placeholders = extract_dynamic_placeholders(query_to_process, nlp)
                # Generate response
                response_gpt = generate_response(model, tokenizer, query_to_process)
                # Replace placeholders
                full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                # time.sleep(1) # Keep optional delay if needed for visual effect

            # Update the placeholder with the final response
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        # Add Assistant message to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        last_role_displayed = "assistant" # Update display tracker

        # Rerun to ensure the history update is reflected smoothly *after* processing
        st.rerun()


    # --- Reset Button ---
    # Place it at the end, outside the main processing flow
    if st.session_state.chat_history:
         st.markdown("---") # Visual separator before reset button
         if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            st.session_state.processing_query = None # Clear any pending query on reset
            st.rerun() # Rerun to clear the chat display
