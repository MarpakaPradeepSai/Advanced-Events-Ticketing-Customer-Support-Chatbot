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

# --- Configuration ---
MODEL_CACHE_DIR = "/tmp/DistilGPT2_Model" # Use /tmp for better compatibility with cloud deployments

# --- Model Loading Functions ---

# Function to download model files from GitHub
def download_model_files(model_dir=MODEL_CACHE_DIR):
    """Downloads model files if they don't exist locally."""
    os.makedirs(model_dir, exist_ok=True)
    all_files_exist = True
    files_to_download = []

    # Check which files already exist
    for filename in MODEL_FILES:
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(local_path):
            all_files_exist = False
            files_to_download.append(filename)

    if all_files_exist:
        # st.info("Model files already downloaded.")
        return True

    st.info(f"Downloading model files to {model_dir}...")
    progress_bar = st.progress(0)
    total_files = len(files_to_download)

    for i, filename in enumerate(files_to_download):
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            progress_bar.progress((i + 1) / total_files)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download {filename} from GitHub: {e}")
            # Clean up partially downloaded file if error occurs
            if os.path.exists(local_path):
                os.remove(local_path)
            return False
    st.success("Model files downloaded successfully.")
    return True

# Load spaCy model for NER
@st.cache_resource(show_spinner="Loading NLP tools...")
def load_spacy_model():
    """Loads the spaCy model, attempting download if necessary."""
    model_name = "en_core_web_trf"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        st.info(f"SpaCy model '{model_name}' not found. Downloading...")
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
            st.success(f"SpaCy model '{model_name}' downloaded and loaded.")
        except Exception as e:
            st.error(f"Failed to download or load spaCy model '{model_name}': {e}")
            st.error("NER functionality will be limited.")
            return None
    return nlp

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading Chatbot Model...")
def load_model_and_tokenizer():
    """Downloads (if needed) and loads the DistilGPT2 model and tokenizer."""
    if not download_model_files(MODEL_CACHE_DIR):
        st.error("Model download failed. Cannot load the model.")
        return None, None
    try:
        model = GPT2LMHeadModel.from_pretrained(MODEL_CACHE_DIR, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_CACHE_DIR)
        # Set pad_token_id if it's not already set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {MODEL_CACHE_DIR}: {e}")
        return None, None

# --- Placeholder Definitions ---
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
    "{{CONTACT_SUPPORT_LINK}}": "<a href='#' target='_blank'>www.support-team.com</a>", # Make links clickable
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
    "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Corrected typo
    "{{SUPPORT_CONTACT_LINK}}": "<a href='#' target='_blank'>www.support-team.com</a>", # Make links clickable
    "{{SUPPORT_SECTION}}": "<b>Support</b>",
    "{{SUPPORT_TEAM_LINK}}": "<a href='#' target='_blank'>www.support-team.com</a>", # Make links clickable
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
    "{{WEBSITE_URL}}": "<a href='#' target='_blank'>www.events-ticketing.com</a>" # Make links clickable
}

# --- Core Logic Functions ---

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    """Replaces static and dynamic placeholders in the response text."""
    # Replace static placeholders first
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    # Then replace dynamic placeholders
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    """Extracts EVENT and GPE (City) entities using spaCy NER."""
    dynamic_placeholders = {}
    if nlp is None: # Handle case where spaCy model failed to load
        st.warning("SpaCy model not loaded. Cannot extract dynamic placeholders like Event/City.", icon="⚠️")
        dynamic_placeholders['{{EVENT}}'] = "the event" # Default fallback
        dynamic_placeholders['{{CITY}}'] = "your city" # Default fallback
        return dynamic_placeholders

    doc = nlp(user_question)
    event_found = False
    city_found = False
    for ent in doc.ents:
        # Prioritize EVENT if found
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
            event_found = True
        # Use GPE (Geopolitical Entity) for city if EVENT is not GPE
        elif ent.label_ == "GPE" and not city_found:
             # Check if the GPE entity text is likely an event name (e.g., contains 'Festival', 'Concert')
             # This is a simple heuristic and might need refinement
            if not any(keyword in ent.text.lower() for keyword in ['festival', 'concert', 'show', 'conference', 'summit', 'expo']):
                city_text = ent.text.title()
                dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
                city_found = True
            # If GPE might be an event and no EVENT tag was found, use it as event
            elif not event_found:
                 event_text = ent.text.title()
                 dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                 event_found = True


    # Provide default values if entities are not found
    if not event_found:
        dynamic_placeholders['{{EVENT}}'] = "the event"
    if not city_found:
        dynamic_placeholders['{{CITY}}'] = "your city"

    # print(f"DEBUG: User Query: '{user_question}' -> Dynamic Placeholders: {dynamic_placeholders}") # Debugging line
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    """Generates a response using the loaded GPT-2 model."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Format the input specifically for the fine-tuned model
    input_text = f"Instruction: {instruction}\nResponse:"

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length // 2).to(device) # Truncate input if too long

    generation_kwargs = {
        "max_length": max_length,
        "num_return_sequences": 1,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id, # Ensure pad_token_id is set
        "eos_token_id": tokenizer.eos_token_id,
        "attention_mask": inputs["attention_mask"]
    }

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                **generation_kwargs
            )
        # Decode the output, skipping special tokens and the prompt
        full_decoded_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Find the start of the actual response after "Response:"
        response_start_tag = "Response:"
        response_start_index = full_decoded_response.find(response_start_tag)

        if response_start_index != -1:
            # Extract the text after "Response:"
            response_text = full_decoded_response[response_start_index + len(response_start_tag):].strip()
            # print(f"DEBUG: Generated raw response: {response_text}") # Debugging line
            return response_text
        else:
            # Fallback if "Response:" tag is not found (might happen with edge cases)
            # Try to remove the input instruction part if possible
            if full_decoded_response.startswith(input_text.replace('\nResponse:','').strip()):
                 response_text = full_decoded_response[len(input_text.replace('\nResponse:','').strip()):].strip()
                 # print(f"DEBUG: Generated raw response (fallback): {response_text}") # Debugging line
                 return response_text
            else:
                # If we can't reliably remove the prompt, return the full decoded output (minus prompt if possible)
                # This might include the instruction part, but it's better than nothing.
                # print(f"DEBUG: Could not find 'Response:' tag. Returning best guess: {full_decoded_response}") # Debugging line
                return full_decoded_response # Or potentially raise an error or return a default message

    except Exception as e:
        st.error(f"Error during model generation: {e}")
        return "Sorry, I encountered an error while generating the response."

# --- CSS Styling ---
st.markdown(
    """
<style>
    /* General Font */
    * {
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #ff8a00, #e52e71); /* Default button gradient */
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 1.1em; /* Slightly adjusted */
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

    /* Specific styling for the 'Ask this question' button */
    /* Targets the first button within a horizontal block layout */
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #29ABE2, #0077B6); /* Blue gradient */
        color: white !important;
    }
     /* Specific styling for the 'Reset Chat' button */
    /* Targets the button with a specific key or relies on position if needed */
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button { /* Assuming Reset is in a vertical block at the end */
        background: linear-gradient(90deg, #6c757d, #343a40); /* Grey gradient */
        color: white !important;
    }
    /* If Reset button isn't easily targetable, you might need a more complex selector or adjust layout */


    /* Chat Input Styling */
    /* Target the textarea within the chat input component */
    div[data-testid="stChatInput"] textarea {
        border: 2px solid black !important; /* Default black border, slightly thicker */
        border-radius: 8px !important; /* Rounded corners */
        background-color: #f0f2f6 !important; /* Light background */
        color: #333 !important; /* Darker text color */
        font-family: 'Times New Roman', Times, serif !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important; /* Smooth transition */
    }

    /* Target the textarea when it has focus (clicked/typing) */
    div[data-testid="stChatInput"] textarea:focus {
        border: 2px solid red !important; /* Red border on focus */
        box-shadow: 0 0 8px rgba(255, 0, 0, 0.5) !important; /* Optional red glow */
        outline: none !important; /* Remove default browser outline */
        background-color: #ffffff !important; /* White background on focus */
    }

    /* Placeholder text styling (optional) */
     div[data-testid="stChatInput"] textarea::placeholder {
        color: #555 !important; /* Darker placeholder text */
        font-family: 'Times New Roman', Times, serif !important;
    }


    /* Horizontal Line Separator */
    .horizontal-line {
        border-top: 1px solid #ccc; /* Thinner grey line */
        margin: 10px 0; /* Adjust spacing */
    }

    /* Streamlit Specific Elements Font */
    .stSelectbox > div > div > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stChatMessage,
    .stAlert, /* Covers st.error, st.warning, st.info, st.success */
    .streamlit-expanderHeader,
    .streamlit-expanderContent {
        font-family: 'Times New Roman', Times, serif !important;
    }

     /* Disclaimer Box Styling */
    .disclaimer-box {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        color: #721c24;
        border: 1px solid #f5c6cb;
        font-family: 'Times New Roman', Times, serif !important; /* Ensure font */
        margin-bottom: 20px; /* Add space below disclaimer */
    }
    .disclaimer-box h1 {
        font-size: 30px; /* Adjusted size */
        color: #721c24;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
         font-family: 'Times New Roman', Times, serif !important; /* Ensure font */
    }
    .disclaimer-box p, .disclaimer-box ul {
        font-size: 16px;
        line-height: 1.6;
        color: #721c24;
         font-family: 'Times New Roman', Times, serif !important; /* Ensure font */
    }
     .disclaimer-box ul {
        margin-left: 20px; /* Indent list */
        list-style-type: disc; /* Use standard bullets */
     }
      .disclaimer-box b { /* Style bold text within disclaimer */
        font-weight: bold;
        color: #5f171e; /* Slightly darker red for emphasis */
     }

    /* Center the Continue button below the disclaimer */
    .continue-button-container {
        display: flex;
        justify-content: flex-end; /* Align button to the right */
        margin-top: 15px;
    }


</style>
""",
    unsafe_allow_html=True,
)

# --- Streamlit UI ---

st.markdown("<h1 style='font-size: 38px; text-align: center;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_role" not in st.session_state:
    st.session_state.last_role = None

# --- Disclaimer Section ---
if not st.session_state.show_chat:
    st.markdown(
        """
        <div class="disclaimer-box">
            <h1>⚠️ Disclaimer</h1>
            <p>
                This <b>Chatbot</b> is designed to assist with ticketing inquiries. However, due to computational limits, it's fine-tuned on specific intents and may not accurately respond to all queries.
            </p>
            <p>
                The chatbot is optimized for:
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
                Assistance for queries outside these areas may be limited.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Container to help align the button
    with st.container():
        st.markdown('<div class="continue-button-container">', unsafe_allow_html=True)
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- Main Chat Interface ---
if st.session_state.show_chat:

    # --- Load Models (only when chat is shown) ---
    # Initialize spaCy model for NER
    nlp = load_spacy_model() # Returns None if loading fails

    # Load DistilGPT2 model and tokenizer
    # This will show the spinner message defined in the @st.cache_resource decorator
    model, tokenizer = load_model_and_tokenizer()

    if model is None or tokenizer is None:
        st.error("Critical Error: Chatbot model could not be loaded. Please check logs or try again later.")
        st.stop() # Stop execution if model loading fails

    st.info("Ask me about ticket cancellations, refunds, upcoming events, or other ticketing questions!", icon="💡")

    # --- Example Queries Section (Dropdown + Button) ---
    example_queries = [
        "How do I buy a ticket?",
        "How can I upgrade my ticket for the upcoming concert in London?", # Added specific example
        "How do I change my personal details on my ticket?",
        "How can I find details about upcoming events in New York?", # Added specific example
        "How do I contact customer service?",
        "How do I get a refund?",
        "What is the ticket cancellation fee?",
        "Can I sell my ticket?"
    ]

    # Use columns for better layout of dropdown and button
    col1, col2 = st.columns([3, 1]) # Adjust ratio as needed
    with col1:
        selected_query = st.selectbox(
            "Or choose an example question:",
            options=[""] + example_queries, # Add an empty option
            index=0, # Default to empty
            key="query_selectbox",
            label_visibility="collapsed" # Hide the label itself
        )
    with col2:
        # Button is only active if a query is selected
        process_query_button = st.button(
            "Ask this question",
            key="query_button",
            disabled=(selected_query == "") # Disable if no query selected
            )

    # --- Chat History Display ---
    st.markdown("---") # Visual separator
    for message in st.session_state.chat_history:
        # Add separator only between user and assistant messages, not between consecutive messages of the same role
        if message["role"] == "user" and st.session_state.last_role == "assistant":
             st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        st.session_state.last_role = message["role"] # Update last role after displaying


    # --- Process Dropdown Selection ---
    if process_query_button and selected_query:
        prompt_from_dropdown = selected_query
        # Capitalize first letter (optional, for consistency)
        prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

        # Add user message to history and display
        st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})
        st.session_state.last_role = "user"
        # Rerun to display the new user message immediately
        st.rerun()


    # --- Handle User Input and Generate Response ---
    # This section runs AFTER potential dropdown processing and rerun
    if prompt := st.chat_input("Enter your own question here..."):
        prompt = prompt.strip()
        if not prompt:
            st.toast("⚠️ Please enter a question.", icon="✍️")
        else:
             # Capitalize first letter (optional)
            prompt = prompt[0].upper() + prompt[1:] if prompt else prompt
            # Add user message to history and display
            st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
            st.session_state.last_role = "user"
            # Rerun to display the new user message immediately
            st.rerun() # Rerun to show the user's message before processing


    # --- Generation Logic (runs if the last message was from the user) ---
    # Check if the last message in history is from the user and needs a response
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        last_user_prompt = st.session_state.chat_history[-1]["content"]

        # Display thinking indicator and generate response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            generating_text = "Thinking..."
            message_placeholder.markdown(f"{generating_text}▌") # Initial placeholder

            start_time = time.time()
            with st.spinner(generating_text): # Official spinner
                # 1. Extract dynamic placeholders (Event, City)
                dynamic_placeholders = extract_dynamic_placeholders(last_user_prompt, nlp)

                # 2. Generate response from the model
                raw_response = generate_response(model, tokenizer, last_user_prompt)

                # 3. Replace placeholders in the generated response
                full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)

            end_time = time.time()
            generation_time = end_time - start_time
            # print(f"DEBUG: Response generation time: {generation_time:.2f} seconds") # Debugging line

            # Update the placeholder with the final response
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        st.session_state.last_role = "assistant"
        # No rerun here needed as the response is already displayed


    # --- Reset Button ---
    # Display reset button only if there's history
    if st.session_state.chat_history:
        st.markdown("---") # Separator before reset button
        if st.button("Reset Chat", key="reset_button"):
            # Clear history and reset state
            st.session_state.chat_history = []
            st.session_state.last_role = None
            # Clear the selectbox selection as well
            st.session_state.query_selectbox = "" # Reset selectbox selection
            st.rerun()
