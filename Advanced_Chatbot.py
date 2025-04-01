# -*- coding: utf-8 -*-
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
    download_successful = True # Flag to track download status
    progress_bar = st.progress(0)
    status_text = st.empty()

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
                        f.write(data)
                        downloaded_size += len(data)
                        # Optional: Update progress more granularly if needed
                        # progress = min(1.0, downloaded_size / total_size) if total_size > 0 else 0
                        # status_text.text(f"Downloading {filename}: {int(progress * 100)}%")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from {url}. Error: {e}")
                download_successful = False
                if os.path.exists(local_path): # Clean up partially downloaded file
                    os.remove(local_path)
                break # Stop downloading if one file fails
            except Exception as e:
                st.error(f"An error occurred while saving {filename}: {e}")
                download_successful = False
                if os.path.exists(local_path):
                     os.remove(local_path)
                break
        else:
             status_text.text(f"{filename} already exists. Skipping.")

        # Update overall progress bar
        progress_bar.progress((i + 1) / len(MODEL_FILES))

    if download_successful:
        status_text.text("Model files downloaded successfully!")
        time.sleep(1) # Give user time to read success message
    else:
        status_text.error("Model download failed. Please check logs and try again.")

    status_text.empty() # Clear status text
    progress_bar.empty() # Clear progress bar
    return download_successful


# Load spaCy model for NER
# Use st.cache_data for models that don't change often
@st.cache_data
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        st.warning("Downloading spaCy model 'en_core_web_trf'...")
        spacy.cli.download("en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")
    return nlp

# Load the DistilGPT2 model and tokenizer
# Use st.cache_resource for resources like models/tokenizers
@st.cache_resource(show_spinner="Loading AI Model...") # Improved spinner message
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot load model. Check your internet connection or GitHub URL.")
        return None, None
    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        st.error("The downloaded model files might be corrupted or incompatible. Try deleting the '/tmp/DistilGPT2_Model' directory and restarting.")
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
    "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note: Space before SECTION might be intentional or a typo
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
    # Replace static placeholders first
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    # Replace dynamic placeholders next
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    if nlp is None: # Add check in case spaCy model failed to load
        st.warning("SpaCy model not loaded. Cannot extract dynamic entities.")
        return {'{{EVENT}}': "event", '{{CITY}}': "city"} # Return defaults

    doc = nlp(user_question)
    dynamic_placeholders = {}
    # Use a flag to track if a specific entity type was found
    event_found = False
    city_found = False

    for ent in doc.ents:
        # Prioritize more specific labels if available (like EVENT over GPE for multi-word entities)
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
            event_found = True
        elif ent.label_ == "GPE" and not city_found: # GPE (Geopolitical Entity) often includes cities
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
            city_found = True
        # Add more entity types if needed (e.g., DATE, TIME, ORG)
        # elif ent.label_ == "DATE":
        #     dynamic_placeholders['{{DATE}}'] = f"<b>{ent.text}</b>"

    # Set default values only if the specific entity wasn't found
    if not event_found:
        dynamic_placeholders['{{EVENT}}'] = "event" # Default placeholder text
    if not city_found:
        dynamic_placeholders['{{CITY}}'] = "city" # Default placeholder text

    return dynamic_placeholders


# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    if model is None or tokenizer is None:
        return "Sorry, the AI model is not available right now."

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Format the input consistently
    input_text = f"Instruction: {instruction} Response:"
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length).to(device) # Added truncation

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,             # Max length of the *generated* sequence
                num_return_sequences=1,
                temperature=0.7,                   # Controls randomness (lower = more deterministic)
                top_p=0.95,                        # Nucleus sampling (considers top p% probability mass)
                do_sample=True,                    # Enable sampling
                pad_token_id=tokenizer.eos_token_id # Prevent generation beyond EOS
            )

        # Decode the generated part only
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True) # Decode only new tokens
        return response.strip()

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, I encountered an error while generating the response."


# --- Streamlit UI Setup ---

st.set_page_config(layout="wide") # Use wider layout

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
    font-size: 1.1em; /* Slightly smaller for better fit */
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: 10px; /* Increased margin */
    margin-bottom: 10px; /* Added margin */
    width: auto;
    min-width: 120px; /* Adjusted min-width */
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

/* Specific Styling for 'Ask this question' Button */
/* Targets the button within the horizontal block likely containing the selectbox */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #29ABE2, #0077B6) !important; /* Ensure override */
    color: white !important;
    font-size: 1.0em; /* Match selectbox better */
    padding: 8px 18px; /* Adjust padding */
    min-width: 150px; /* Ensure it fits text */
    margin-top: 0px; /* Align with selectbox */
    margin-bottom: 0px;
}

/* Styling for the 'Reset Chat' Button */
/* Target button specifically by key or position if possible, otherwise general .stButton applies */
/* If Reset button needs unique style, give it a specific class or use nth-of-type if layout is stable */


/* Apply Times New Roman to all text elements */
body, .stApp * {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Ensure selectbox text uses the font */
.stSelectbox div[data-baseweb="select"] > div {
     font-family: 'Times New Roman', Times, serif !important;
}
/* Ensure text input/area uses the font */
.stTextInput input, .stTextArea textarea {
    font-family: 'Times New Roman', Times, serif !important;
}
/* Ensure chat message content uses the font */
.stChatMessage {
    font-family: 'Times New Roman', Times, serif !important;
}
/* Ensure error/warning messages use the font */
.stAlert { /* More general class for alerts */
    font-family: 'Times New Roman', Times, serif !important;
}

/* Horizontal Line Separator */
.horizontal-line {
    border-top: 2px solid #e0e0e0;
    margin: 20px 0; /* Increased spacing */
}

/* Disclaimer Box Styling */
.disclaimer-box {
    background-color: #f8d7da;
    padding: 25px; /* Increased padding */
    border-radius: 10px;
    color: #721c24;
    border: 1px solid #f5c6cb;
    margin-bottom: 20px; /* Space below disclaimer */
}
.disclaimer-box h1 {
    font-size: 32px; /* Adjusted size */
    color: #721c24;
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
}
.disclaimer-box p, .disclaimer-box ul {
    font-size: 16px;
    line-height: 1.6;
    color: #721c24;
}
.disclaimer-box ul {
    margin-left: 20px; /* Indent list */
    list-style-type: disc; /* Use standard bullets */
}

/* Container for the Continue Button */
.continue-button-container {
    text-align: right; /* Align content (the button) to the right */
    width: 100%; /* Ensure the container spans the width */
    margin-top: 15px; /* Add some space above the button */
}
/* Specific styling for the Continue button if needed (inherits .stButton>button) */
.continue-button-container .stButton>button {
     background: linear-gradient(90deg, #28a745, #218838); /* Green gradient for Continue */
     min-width: 130px;
}


</style>
    """,
    unsafe_allow_html=True,
)


# --- Main App Logic ---

st.markdown("<h1 style='font-size: 40px; text-align: center; margin-bottom: 20px;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "nlp_model" not in st.session_state:
    st.session_state.nlp_model = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None


# --- Disclaimer Section ---
if not st.session_state.show_chat:
    st.markdown(
        """
        <div class="disclaimer-box">
            <h1>⚠️ Disclaimer</h1>
            <p>
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents and may not be able to respond accurately to all types of queries.
            </p>
            <p>
                The chatbot is optimized to handle the following intents:
            </p>
            <ul>
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
            <p>
                Please note that this chatbot may not be able to assist with queries outside of these predefined intents.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Continue button aligned to the right using the container div
    st.markdown('<div class="continue-button-container">', unsafe_allow_html=True)
    if st.button("Continue", key="continue_button"):
        # Load models only when user clicks continue
        with st.spinner("Initializing models, please wait..."):
            st.session_state.nlp_model = load_spacy_model()
            st.session_state.llm_model, st.session_state.tokenizer = load_model_and_tokenizer()

        if st.session_state.llm_model and st.session_state.tokenizer and st.session_state.nlp_model:
            st.session_state.show_chat = True
            st.rerun() # Rerun to hide disclaimer and show chat
        else:
            st.error("Failed to initialize necessary models. Cannot continue.")
            # Keep show_chat as False
    st.markdown('</div>', unsafe_allow_html=True)

# --- Chat Interface Section ---
elif st.session_state.show_chat:
    # Ensure models are loaded (safety check)
    if not st.session_state.llm_model or not st.session_state.tokenizer or not st.session_state.nlp_model:
        st.error("Models are not loaded correctly. Please reload the page or contact support.")
        st.stop() # Stop execution if models aren't ready

    # Retrieve models from session state
    nlp = st.session_state.nlp_model
    model = st.session_state.llm_model
    tokenizer = st.session_state.tokenizer

    st.info("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # Example queries dropdown and button
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

    # Use columns for better alignment of selectbox and button
    col1, col2 = st.columns([3, 1]) # Adjust ratio as needed
    with col1:
        selected_query = st.selectbox(
            "Choose a query from examples or type your own below:",
            ["Choose your question"] + example_queries,
            key="query_selectbox",
            label_visibility="collapsed" # Hide label for cleaner look
        )
    with col2:
        process_query_button = st.button("Ask selected", key="query_button") # Shorter button text

    # --- Chat History Display ---
    chat_container = st.container() # Use a container for chat messages
    with chat_container:
        last_role = None
        for i, message in enumerate(st.session_state.chat_history):
            is_last_message = (i == len(st.session_state.chat_history) - 1)
            # Add separator line between user/assistant pairs, but not before the very first message
            if message["role"] == "user" and last_role == "assistant" and i > 0:
                 st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"], unsafe_allow_html=True)
            last_role = message["role"]

            # Don't add a line after the very last message
            # if message["role"] == "assistant" and not is_last_message:
            #      st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)


    # --- Handle User Input (Dropdown or Text Input) ---
    user_input = None
    input_source = None # To track where the input came from

    # Process selected query from dropdown
    if process_query_button and selected_query != "Choose your question":
        user_input = selected_query
        input_source = "dropdown"

    # Process text input
    if prompt := st.chat_input("Enter your own question here:"):
        if prompt.strip():
            user_input = prompt
            input_source = "text_input"
        else:
            st.toast("⚠️ Please enter a question.", icon="⚠️")

    # --- Generate and Display Response ---
    if user_input:
        # Capitalize first letter (optional, stylistic choice)
        processed_input = user_input[0].upper() + user_input[1:] if user_input else user_input

        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": processed_input, "avatar": "👤"})

        # Rerun to display the user's message immediately
        # This makes the UI feel more responsive before the bot generates
        st.rerun()


# --- Handle Response Generation (needs to run after potential rerun) ---
# Check if the last message was from the user, indicating a response is needed
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
     last_user_message = st.session_state.chat_history[-1]["content"]

     # Check models again before generation
     if not st.session_state.llm_model or not st.session_state.tokenizer or not st.session_state.nlp_model:
        st.error("Models not loaded. Cannot generate response.")
        st.stop()

     nlp = st.session_state.nlp_model
     model = st.session_state.llm_model
     tokenizer = st.session_state.tokenizer

     # Display thinking indicator and generate response
     with chat_container: # Add the thinking indicator within the chat container
         with st.chat_message("assistant", avatar="🤖"):
             message_placeholder = st.empty()
             generating_text = "Thinking..."
             message_placeholder.markdown(f"{generating_text}▌") # Use cursor effect

             try:
                 dynamic_placeholders = extract_dynamic_placeholders(last_user_message, nlp)
                 raw_response = generate_response(model, tokenizer, last_user_message)
                 full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)

                 # Simulate typing effect (optional)
                 # response_so_far = ""
                 # for word in full_response.split():
                 #     response_so_far += word + " "
                 #     message_placeholder.markdown(f"{response_so_far}▌", unsafe_allow_html=True)
                 #     time.sleep(0.05) # Adjust speed as needed

                 message_placeholder.markdown(full_response, unsafe_allow_html=True) # Display final response

                 # Add assistant response to history
                 st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})

             except Exception as e:
                 st.error(f"An error occurred while generating the response: {e}")
                 error_message = "Sorry, I encountered an issue. Please try again."
                 message_placeholder.markdown(error_message, unsafe_allow_html=True)
                 # Add error message to history (optional)
                 st.session_state.chat_history.append({"role": "assistant", "content": error_message, "avatar": "🤖"})

     # Rerun *after* adding the assistant message to update the display fully
     st.rerun()


# --- Reset Button ---
if st.session_state.chat_history:
    st.markdown("---") # Add a visual separator before the reset button
    if st.button("Reset Chat", key="reset_button"):
        st.session_state.chat_history = []
        # Optionally clear selected query
        st.session_state.query_selectbox = "Choose your question"
        st.rerun()
