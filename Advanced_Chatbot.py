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
    all_files_exist = True
    for filename in MODEL_FILES:
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(local_path):
            all_files_exist = False
            break

    if all_files_exist:
        # print("All model files already exist locally.")
        return True

    # If any file is missing, download all
    # print("Downloading model files...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, filename in enumerate(MODEL_FILES):
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

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
                    # Update progress within the loop if needed (though updating per file is simpler)

            status_text.text(f"Downloaded {filename}...")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download {filename} from GitHub: {e}")
            # Clean up partially downloaded file if error occurs
            if os.path.exists(local_path):
                os.remove(local_path)
            # Clean up the entire directory if one file fails
            for f_to_remove in MODEL_FILES:
                 path_to_remove = os.path.join(model_dir, f_to_remove)
                 if os.path.exists(path_to_remove):
                     os.remove(path_to_remove)
            if os.path.exists(model_dir):
                 try:
                     os.rmdir(model_dir) # Only removes if empty, but useful after cleaning files
                 except OSError:
                     pass # Ignore if not empty (shouldn't happen after removing files)
            progress_bar.empty() # Remove progress bar on failure
            status_text.empty() # Remove status text
            return False

        # Update progress bar after each file download
        progress = (i + 1) / len(MODEL_FILES)
        progress_bar.progress(progress)

    status_text.text("Model download complete!")
    time.sleep(1) # Keep completion message visible briefly
    progress_bar.empty()
    status_text.empty()
    return True


# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        # Try loading the transformer model first
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        st.warning("Transformer model 'en_core_web_trf' not found. Downloading...")
        try:
            spacy.cli.download("en_core_web_trf")
            nlp = spacy.load("en_core_web_trf")
        except Exception as e:
            st.error(f"Failed to download or load 'en_core_web_trf': {e}")
            st.warning("Falling back to 'en_core_web_sm'. NER might be less accurate.")
            try:
                 # Try loading the small model as a fallback
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                st.warning("Small model 'en_core_web_sm' not found. Downloading...")
                try:
                    spacy.cli.download("en_core_web_sm")
                    nlp = spacy.load("en_core_web_sm")
                except Exception as e_sm:
                    st.error(f"Failed to download or load 'en_core_web_sm': {e_sm}")
                    st.error("No suitable spaCy model could be loaded. NER functionality will be limited.")
                    return None # Return None if no model can be loaded
    return nlp


# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading Chat Model...")
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    # Trigger download within the cached function if needed
    if not download_model_files(model_dir):
        st.error("Model download failed. Chatbot cannot start.")
        return None, None

    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        st.error("Please ensure all model files were downloaded correctly and are not corrupted.")
         # Attempt to clean up potentially corrupted files
        st.warning(f"Attempting to clear corrupted model cache at {model_dir}...")
        for filename in MODEL_FILES:
             local_path = os.path.join(model_dir, filename)
             if os.path.exists(local_path):
                 try:
                     os.remove(local_path)
                 except OSError as rm_err:
                     st.error(f"Could not remove {local_path}: {rm_err}")
        if os.path.exists(model_dir):
             try:
                 os.rmdir(model_dir)
             except OSError as rmdir_err:
                 st.error(f"Could not remove directory {model_dir}: {rmdir_err}")
        st.warning("Cache cleared (if possible). Please try rerunning the application to redownload the model.")
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
    "{{SUPPORT_ SECTION}}": "<b>Support</b>",
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
    dynamic_placeholders = {}
    if nlp is None: # Handle case where spaCy model failed to load
        st.warning("spaCy model not loaded. Cannot extract dynamic entities like EVENT or CITY.")
        dynamic_placeholders['{{EVENT}}'] = "event"
        dynamic_placeholders['{{CITY}}'] = "city"
        return dynamic_placeholders

    doc = nlp(user_question)

    # Prioritize PERSON if found, otherwise look for EVENT/GPE
    person_found = False
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person_text = ent.text.title()
            # Assuming a person might be an event name in this context
            dynamic_placeholders['{{EVENT}}'] = f"<b>{person_text}</b>"
            person_found = True
            break # Take the first person found

    # If no PERSON, look for EVENT
    if not person_found:
        for ent in doc.ents:
             if ent.label_ == "EVENT":
                event_text = ent.text.title()
                dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                break # Take the first event found

    # Look for GPE (City/Location) regardless of PERSON/EVENT
    for ent in doc.ents:
        if ent.label_ == "GPE":
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
            break # Take the first city/location found

    # Set defaults if placeholders weren't filled
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "the event" # More natural default
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "your city" # More natural default

    return dynamic_placeholders


# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length).to(device) # Added truncation and used model_max_length

    # Define generation parameters
    gen_kwargs = {
        "max_length": max_length,
        "num_return_sequences": 1,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "attention_mask": inputs["attention_mask"] # Pass attention mask explicitly
    }

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                **gen_kwargs
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Find the start of the response part
        response_start_marker = "Response:"
        response_start_index = response.find(response_start_marker)

        if response_start_index != -1:
            # Extract text after "Response:"
            response_text = response[response_start_index + len(response_start_marker):].strip()
        else:
            # Fallback if "Response:" marker is not found (might happen with short generations)
            # Attempt to remove the instruction part if possible
            if response.startswith(input_text.replace(" Response:", "")):
                 response_text = response[len(input_text.replace(" Response:", "")):].strip()
            else:
                 response_text = response # Return the whole output as fallback

        # Basic check for incomplete or placeholder-like responses
        if not response_text or response_text == instruction:
            return "I'm sorry, I couldn't generate a specific response for that. Could you please rephrase your question?"

        return response_text

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "I encountered an error while trying to generate a response. Please try again."


# --- CSS Styling ---
st.markdown(
    """
<style>
/* Button Styling */
.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71); /* Stylish gradient */
    color: white !important; /* Ensure text is white */
    border: none;
    border-radius: 25px; /* Rounded corners */
    padding: 10px 20px; /* Padding */
    font-size: 1.1em; /* Slightly adjusted font size */
    font-weight: bold; /* Bold text */
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease; /* Smooth transitions */
    display: inline-flex; /* Helps with alignment */
    align-items: center;
    justify-content: center;
    margin-top: 5px;
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

/* General Font Styling */
body, .stApp, .stChatInput, .stChatMessage, .stSelectbox, .stTextInput, .stTextArea, .stAlert, .stMarkdown {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Specific adjustments for Streamlit elements */
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
.stChatMessage p { /* Ensure paragraphs within chat messages also use the font */
    font-family: 'Times New Roman', Times, serif !important;
}
.st-emotion-cache-r421ms { /* Example class for st.error, st.warning, etc. */
    font-family: 'Times New Roman', Times, serif !important;
}
.streamlit-expanderContent p { /* For text inside expanders if used */
    font-family: 'Times New Roman', Times, serif !important;
}

/* Styling for the "Ask this question" button specifically */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */
    color: white !important;
    font-size: 1.0em; /* Slightly smaller font for this button */
    padding: 8px 16px; /* Slightly smaller padding */
}

/* Horizontal line separator */
.horizontal-line {
    border-top: 1px solid #ccc; /* Thinner, lighter grey line */
    margin: 10px 0; /* Adjust spacing */
}

/* --- Style for Chat Input Area Shadow --- */
div[data-testid="stChatInput"] {
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15); /* Adjust values for desired shadow */
    border-radius: 10px; /* Optional: rounds the corners of the input area */
    padding: 5px 10px; /* Optional: Add some padding around the input itself */
    margin: 10px 0; /* Optional: Add some margin above/below */
    background-color: #ffffff; /* Ensure a solid background color */
    border: 1px solid #eee; /* Optional: Add a subtle border */
}
/* --- End Style for Chat Input Area Shadow --- */

</style>
    """,
    unsafe_allow_html=True,
)
# --- End CSS Styling ---

# --- Streamlit UI ---
st.markdown("<h1 style='font-size: 43px; font-family: \"Times New Roman\", Times, serif;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "nlp_loaded" not in st.session_state:
    st.session_state.nlp_loaded = False
if "last_role" not in st.session_state:
     st.session_state.last_role = None


# Example queries for dropdown
example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the upcoming concert in London?", # Added specific example
    "How do I change my personal details on my ticket?",
    "How can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?",
    "What is the ticket cancellation fee?",
    "Can I sell my ticket?"
]

# --- Disclaimer Screen ---
if not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb; font-family: 'Times New Roman', Times, serif;">
            <h1 style="font-size: 36px; color: #721c24; font-weight: bold; text-align: center; font-family: 'Times New Roman', Times, serif;">⚠️ Disclaimer</h1>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24; font-family: 'Times New Roman', Times, serif;">
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents, and may not be able to respond accurately to all types of queries.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24; font-family: 'Times New Roman', Times, serif;">
                The chatbot is optimized to handle the following intents:
            </p>
            <ul style="font-size: 16px; line-height: 1.6; color: #721c24; list-style-position: inside; padding-left: 20px; font-family: 'Times New Roman', Times, serif;">
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
            <p style="font-size: 16px; line-height: 1.6; color: #721c24; font-family: 'Times New Roman', Times, serif;">
                Please note that this chatbot may not be able to assist with queries outside of these predefined intents. Click 'Continue' to start chatting after models are loaded.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load models while showing disclaimer
    if not st.session_state.nlp_loaded:
        nlp = load_spacy_model()
        if nlp:
            st.session_state.nlp = nlp
            st.session_state.nlp_loaded = True
        else:
             st.error("Failed to load spaCy model. NER features will be unavailable.")
             # Allow continuing without NER, but features will be degraded
             st.session_state.nlp = None # Explicitly set to None
             st.session_state.nlp_loaded = True # Mark as "loaded" (even if null) to proceed


    if not st.session_state.model_loaded and st.session_state.nlp_loaded: # Load GPT model only after spaCy attempt
        model, tokenizer = load_model_and_tokenizer()
        if model and tokenizer:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
        else:
            st.error("Chat model failed to load. The application cannot continue.")
            st.stop() # Stop execution if the main model fails

    # Only show continue button if both models are loaded (or attempted to load)
    if st.session_state.model_loaded and st.session_state.nlp_loaded:
        col1, col2 = st.columns([4, 1])  # Adjust ratios as needed
        with col2:
            if st.button("Continue", key="continue_button"):
                st.session_state.show_chat = True
                st.rerun()
    else:
        st.info("Please wait while the necessary models are being loaded...")


# --- Chat Interface Screen ---
if st.session_state.show_chat:
    # Ensure models are loaded before proceeding (redundancy check)
    if not st.session_state.model_loaded or not st.session_state.nlp_loaded:
        st.error("Models are not loaded correctly. Please reload the page.")
        st.stop()

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    nlp = st.session_state.nlp # Can be None if spaCy failed

    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # --- Dropdown and Button Section ---
    query_col, button_col = st.columns([4, 1]) # Adjust ratio if needed
    with query_col:
        selected_query = st.selectbox(
            "Choose a query from examples:",
            ["Choose your question"] + example_queries,
            key="query_selectbox",
            label_visibility="collapsed" # Hides the label visually but keeps it for accessibility
        )
    with button_col:
        process_query_button = st.button("Ask this", key="query_button") # Shorter button text

    # --- Chat History Display ---
    chat_container = st.container() # Use a container for chat messages
    with chat_container:
        for message in st.session_state.chat_history:
             # Add separator line if the last message was from the assistant
             # This needs to be checked *before* displaying the current user message
             if message["role"] == "user" and st.session_state.last_role == "assistant":
                  st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

             with st.chat_message(message["role"], avatar=message["avatar"]):
                  st.markdown(message["content"], unsafe_allow_html=True)
             st.session_state.last_role = message["role"] # Update last role *after* displaying


    # --- Process Dropdown Query ---
    if process_query_button:
        if selected_query == "Choose your question":
            st.toast("⚠️ Please select a question from the dropdown.", icon="💡")
        elif selected_query:
            prompt_from_dropdown = selected_query
            # No need to capitalize, let user input be as is.
            # prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

            # Add user message to history and display
            st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})

            # Rerun to update the chat display immediately
            st.rerun() # This will handle displaying the new user message and the separator logic

    # --- Process Text Input Query ---
    if prompt := st.chat_input("Enter your own question:"):
        # No need to capitalize
        # prompt = prompt[0].upper() + prompt[1:] if prompt else prompt
        if not prompt.strip():
            st.toast("⚠️ Please enter a question.", icon="✍️")
        else:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
            # Rerun immediately to show the user message
            st.rerun()

    # --- Generate and Display Assistant Response ---
    # Check if the last message in history is from the user and needs a response
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        user_prompt = st.session_state.chat_history[-1]["content"]

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            generating_response_text = "Thinking..." # Shorter spinner text
            full_response = "" # Initialize empty response
            with st.spinner(generating_response_text):
                # Extract dynamic placeholders using spaCy (handles nlp=None case)
                dynamic_placeholders = extract_dynamic_placeholders(user_prompt, nlp)

                # Generate response from the GPT model
                response_gpt = generate_response(model, tokenizer, user_prompt)

                # Replace placeholders in the generated response
                full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                # time.sleep(0.5) # Optional small delay for effect

            # Display the final response
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        # Add assistant response to history *after* displaying it
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        st.session_state.last_role = "assistant" # Update last role *after* adding assistant message
        # No rerun needed here as the response is displayed directly


    # --- Reset Button ---
    if st.session_state.chat_history:
        # Use columns to right-align the reset button
        _, btn_col = st.columns([6, 1]) # Adjust ratio to push button right
        with btn_col:
             if st.button("Reset", key="reset_button"): # Simpler button text
                 st.session_state.chat_history = []
                 st.session_state.last_role = None
                 st.rerun()
