import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time # <--- Import the time module

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
    download_success = True # Flag to track download status

    # Use a placeholder for progress updates
    progress_text = st.empty()
    total_files = len(MODEL_FILES)
    files_downloaded = 0

    for filename in MODEL_FILES:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

        # Check if file exists and is not empty
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            files_downloaded += 1
            progress_text.text(f"Downloading model files... ({files_downloaded}/{total_files}) - {filename}")
            try:
                response = requests.get(url, stream=True, timeout=60) # Add stream and timeout
                response.raise_for_status() # Raise an exception for bad status codes
                
                # Write content chunk by chunk (useful for large files)
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify download size if possible (requires Content-Length header)
                # content_length = response.headers.get('Content-Length')
                # if content_length and os.path.getsize(local_path) != int(content_length):
                #     st.warning(f"Warning: Size mismatch for {filename}. File might be corrupted.")
                #     download_success = False # Optionally mark as failure if size mismatch is critical

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from GitHub: {e}")
                # Attempt to delete potentially incomplete file
                if os.path.exists(local_path):
                    os.remove(local_path)
                download_success = False
                break # Stop downloading if one file fails
        else:
            # If file exists, just update the count for the progress message
            files_downloaded += 1
            progress_text.text(f"Downloading model files... ({files_downloaded}/{total_files}) - {filename} (cached)")


    progress_text.empty() # Clear the progress text
    return download_success


# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        # Check if the model is already downloaded, otherwise download it
        spacy.load("en_core_web_trf")
    except OSError:
        st.info("Downloading spaCy model 'en_core_web_trf'... This may take a moment.")
        spacy.cli.download("en_core_web_trf")
    return spacy.load("en_core_web_trf")

# Load the DistilGPT2 model and tokenizer
# Increase ttl (time-to-live) if needed, default is forever for @st.cache_resource
@st.cache_resource(show_spinner="Downloading and loading language model...")
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Please check your internet connection or the GitHub URL and refresh.")
        return None, None

    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model from disk: {e}. Try clearing cache and refreshing.")
        # Attempt to clear the model directory if loading fails, prompting a redownload next time
        # Be cautious with this in production environments
        # import shutil
        # if os.path.exists(model_dir):
        #     shutil.rmtree(model_dir)
        #     st.info("Cleared potentially corrupted model cache. Please refresh the page.")
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
    # Replace dynamic first to avoid conflicts if dynamic values contain static placeholders
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    doc = nlp(user_question)
    dynamic_placeholders = {}
    event_found = False
    city_found = False
    for ent in doc.ents:
        # Prioritize specific entity labels if available, or use common ones like ORG for event as fallback
        if ent.label_ in ["EVENT", "ORG", "WORK_OF_ART"]: # Add relevant labels
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
            event_found = True
        elif ent.label_ == "GPE": # Geopolitical Entity (cities, states, countries)
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
            city_found = True
        # Add more entity types if needed (e.g., DATE, TIME)

    # Provide defaults if not found
    if not event_found:
        dynamic_placeholders['{{EVENT}}'] = "the event" # More generic default
    if not city_found:
        dynamic_placeholders['{{CITY}}'] = "your city" # More generic default
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Ensure model is on the correct device

    # Format the input correctly based on how the model was fine-tuned
    # Assuming the fine-tuning used "Instruction: ... Response:" format
    input_text = f"Instruction: {instruction} Response:"
    
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length).to(device)

    # Generate response
    try:
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
        
        # Decode the generated tokens, skipping special tokens and the input prompt part
        # Find the start of the response after the prompt
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, I encountered an error while generating the response."


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
    font-size: 1.2em; /* Font size */
    font-weight: bold; /* Bold text */
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease; /* Smooth transitions */
    display: inline-flex; /* Helps with alignment */
    align-items: center;
    justify-content: center;
    margin-top: 5px; /* Adjust slightly if needed for alignment */
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

/* Specific styling for the 'Ask this question' button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */
    color: white !important;
}

/* Specific styling for the 'Reset Chat' button */
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button[kind="secondary"] { /* Target reset button more specifically if possible */
    background: linear-gradient(90deg, #6c757d, #343a40) !important; /* Grey gradient */
    color: white !important;
    border: 1px solid #6c757d !important; /* Add a subtle border */
}
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button[kind="secondary"]:hover {
    box-shadow: 0px 3px 10px rgba(0, 0, 0, 0.2) !important;
    transform: scale(1.03) !important;
}


/* Apply Times New Roman to all text elements */
body, .stApp, .stMarkdown, .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > div, .stChatMessage p {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Horizontal Line Separator */
.horizontal-line {
    border-top: 2px solid #e0e0e0; /* Adjust color and thickness as needed */
    margin: 15px 0; /* Adjust spacing above and below the line */
}

/* Chat Input Shadow Effect */
div[data-testid="stChatInput"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Softer shadow */
    border-radius: 8px;
    padding: 10px 15px; /* Adjust padding */
    margin: 15px 0;
    background-color: #ffffff; /* Ensure background is white */
    border: 1px solid #eee; /* Subtle border */
}

/* Response Time Styling */
.response-time {
    font-size: 0.8em;
    color: #888;
    margin-top: 5px;
    display: block; /* Ensure it appears on a new line */
}

</style>
    """,
    unsafe_allow_html=True,
)

# --- Streamlit UI ---
st.markdown("<h1 style='font-size: 43px; font-family: \"Times New Roman\", Times, serif;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Model Loading ---
# Load models only once and store in session state
if not st.session_state.models_loaded:
    # Use columns to center the spinner if desired
    col1_load, col2_load, col3_load = st.columns([1,2,1])
    with col2_load:
        with st.spinner("Initializing chatbot... This might take a minute."):
            try:
                nlp = load_spacy_model()
                model, tokenizer = load_model_and_tokenizer()

                if model is not None and tokenizer is not None and nlp is not None:
                    st.session_state.models_loaded = True
                    st.session_state.nlp = nlp
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    # st.success("Models loaded successfully!") # Optional success message
                    # time.sleep(1) # Brief pause to show success
                    st.rerun() # Rerun to proceed past loading phase
                else:
                    # Error messages are handled within the loading functions
                    st.error("Initialization failed. Please refresh the page.") # Generic fallback
                    st.stop() # Stop execution if loading failed critically

            except Exception as e:
                st.error(f"An unexpected error occurred during initialization: {str(e)}")
                st.stop() # Stop execution on unexpected error

# --- Disclaimer Screen ---
# Display Disclaimer and Continue button ONLY if models are loaded but chat hasn't started
if st.session_state.models_loaded and not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb; font-family: 'Times New Roman', Times, serif;">
            <h1 style="font-size: 36px; color: #721c24; font-weight: bold; text-align: center;">⚠️ Disclaimer</h1>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents and may not respond accurately to all query types.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                The chatbot is optimized to handle intents such as:
            </p>
            <ul style="font-size: 16px; line-height: 1.6; color: #721c24; margin-left: 20px;">
                <li>Buying, Selling, Transferring, Upgrading Tickets</li>
                <li>Cancelling Tickets & Checking Fees</li>
                <li>Finding Tickets & Upcoming Events</li>
                <li>Getting Refunds & Tracking Cancellation</li>
                <li>Updating Personal Ticket Details</li>
                <li>General Customer Service Inquiries</li>
                <li>Ticket Information</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                Please note that the chatbot may struggle with queries outside these areas. Your patience is appreciated if the response isn't perfect.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Continue button aligned to the right using columns
    _, col_btn = st.columns([4, 1]) # Adjust ratio if needed
    with col_btn:
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun() # Rerun to show the chat interface

# --- Chat Interface ---
# Show chat interface only after clicking Continue and models are loaded
if st.session_state.models_loaded and st.session_state.show_chat:

    # Access loaded models from session state (ensure they are loaded)
    if 'nlp' not in st.session_state or 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        st.error("Models not loaded correctly. Please refresh.")
        st.stop()
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    st.write("Ask me about ticket cancellations, refunds, events, or other inquiries!")

    # --- Example Queries Section ---
    example_queries = [
        "How do I buy a ticket?",
        "How can I upgrade my ticket for the Music Fest in London?", # Example with entities
        "How do I change my personal details on my ticket?",
        "How can I find details about upcoming events?",
        "How do I contact customer service?",
        "How do I get a refund?",
        "What is the ticket cancellation fee?",
        "How can I track my ticket cancellation status?",
        "How can I sell my ticket?",
        "Tell me about the refund policy."
    ]

    # Use columns for layout of dropdown and button
    col_select, col_button = st.columns([4, 1]) # Adjust ratio as needed

    with col_select:
        selected_query = st.selectbox(
            "Or choose a query from examples:",
            options=[""] + example_queries, # Add empty option as placeholder
            index=0, # Default to empty selection
            key="query_selectbox",
            label_visibility="collapsed" # Hide the label itself
        )

    with col_button:
        process_query_button = st.button("Ask this", key="query_button")

    # --- Chat History Display ---
    last_role = None # Track last message role for separator
    for message in st.session_state.chat_history:
        is_user = message["role"] == "user"
        # Add separator line before user message if previous was assistant
        if is_user and last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

        avatar_symbol = "👤" if is_user else "🤖"
        with st.chat_message(message["role"], avatar=avatar_symbol):
            st.markdown(message["content"], unsafe_allow_html=True)
            # Display response time if it exists in the history (for assistant messages)
            if not is_user and "time_taken" in message:
                 st.markdown(f"<span class='response-time'>Generated in {message['time_taken']:.2f}s</span>", unsafe_allow_html=True)

        last_role = message["role"]

    # --- Handle Query Submission (Dropdown or Input) ---

    # Function to process a query (to avoid code duplication)
    def process_and_display_query(query_text):
        nonlocal last_role # Allow modification of last_role from parent scope

        # Add user message to history and display
        st.session_state.chat_history.append({"role": "user", "content": query_text})
        if last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(query_text, unsafe_allow_html=True)
        last_role = "user"

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Generating response..."):
                start_time = time.time() # <--- Start timer
                
                # Perform NER and generate response
                dynamic_placeholders = extract_dynamic_placeholders(query_text, nlp)
                raw_response = generate_response(model, tokenizer, query_text)
                full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)
                
                end_time = time.time() # <--- End timer
                response_time = end_time - start_time

            # Display response and time
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
            st.markdown(f"<span class='response-time'>Generated in {response_time:.2f}s</span>", unsafe_allow_html=True)

        # Add assistant response (with time) to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "time_taken": response_time # <-- Store time in history
        })
        last_role = "assistant"

    # Process selected query from dropdown
    if process_query_button and selected_query:
        query = selected_query[0].upper() + selected_query[1:] if selected_query else ""
        process_and_display_query(query)
        # Clear the selectbox selection after processing (optional)
        # st.session_state.query_selectbox = "" # Causes rerun, might be slightly jarring
        st.rerun() # Rerun to update display correctly after processing dropdown


    # Process input from chat input box
    if prompt := st.chat_input("Enter your own question:"):
        query = prompt[0].upper() + prompt[1:] if prompt else ""
        if not query.strip():
            st.toast("⚠️ Please enter a question.", icon="❓")
        else:
            process_and_display_query(query)
            st.rerun() # Rerun to update the display after processing input


    # --- Reset Button ---
    # Conditionally display reset button only if there's history
    if st.session_state.chat_history:
        st.markdown("---") # Add a visual separator before the button
        # Use columns to potentially center or right-align the button
        _, col_reset, _ = st.columns([3, 1, 3])
        with col_reset:
           if st.button("Reset Chat", key="reset_button", type="secondary"): # Use secondary type for different styling
                st.session_state.chat_history = []
                last_role = None
                st.rerun() # Rerun to clear the chat display
