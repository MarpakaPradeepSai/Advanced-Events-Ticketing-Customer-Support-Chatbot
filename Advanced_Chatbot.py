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
    st.info(f"Checking/Downloading model files to {model_dir}...")
    all_files_exist = True
    files_to_download = []

    for filename in MODEL_FILES:
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(local_path):
            all_files_exist = False
            files_to_download.append(filename)

    if all_files_exist:
        st.success("All model files already exist locally.")
        return True

    st.warning(f"Downloading missing files: {', '.join(files_to_download)}")
    progress_bar = st.progress(0)
    total_files = len(files_to_download)
    downloaded_count = 0

    for filename in files_to_download:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            total_size = int(response.headers.get('content-length', 0))
            st.write(f"Downloading {filename} ({total_size / (1024 * 1024):.2f} MB)...") # Show file size
            with open(local_path, "wb") as f:
                # Simple progress indication per file (not chunk-based)
                 f.write(response.content)

            downloaded_count += 1
            progress_bar.progress(downloaded_count / total_files)
            st.write(f"Downloaded {filename}")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download {filename} from GitHub: {e}")
            # Clean up partially downloaded file if error occurs
            if os.path.exists(local_path):
                os.remove(local_path)
            return False
        except Exception as e:
            st.error(f"An error occurred while writing {filename}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return False


    progress_bar.empty() # Remove progress bar after completion
    st.success("Model files downloaded successfully.")
    return True


# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_trf"
    try:
        # Check if model is installed
        spacy.load(model_name)
    except OSError:
        st.warning(f"spaCy model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        st.success(f"spaCy model '{model_name}' downloaded successfully.")
    return spacy.load(model_name)

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading AI model...") # Add spinner text
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot proceed.")
        return None, None
    try:
        st.write("Loading Model from local directory...")
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        st.success("AI Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model/tokenizer from {model_dir}: {e}")
        # Attempt to clean up potentially corrupted download
        st.warning("Attempting to clear cached model files...")
        try:
            import shutil
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            st.info("Cleared model directory. Please refresh the page to retry download.")
        except Exception as cleanup_e:
            st.error(f"Could not clear model directory: {cleanup_e}")
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
    doc = nlp(user_question)
    dynamic_placeholders = {}
    # Use descriptive variable names instead of generic 'ent'
    for entity in doc.ents:
        if entity.label_ == "EVENT":
            event_text = entity.text.title() # Capitalize event names
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif entity.label_ == "GPE": # GPE often represents cities/locations
            city_text = entity.text.title() # Capitalize city names
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
    # Provide default values if entities are not found
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
    # Ensure consistent instruction format
    input_text = f"Instruction: {instruction} Response:"
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length).to(device) # Added truncation
    with torch.no_grad():
        # Generate response
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length + inputs["input_ids"].shape[1], # Adjust max_length relative to input
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode response
    response_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response_marker = "Response:"
    response_start_index = response_full.find(response_marker)
    if response_start_index != -1:
        response_text = response_full[response_start_index + len(response_marker):].strip()
    else:
        # Fallback if "Response:" marker isn't found (e.g., if model behaves unexpectedly)
        # Try to find the response after the original instruction
        instruction_marker = "Instruction:"
        instruction_end_index = response_full.find(instruction_marker)
        if instruction_end_index != -1:
             # Find the end of the instruction part
             temp_text = response_full[instruction_end_index + len(instruction_marker):]
             # Assume response starts after the instruction text in the input
             original_instruction_end = temp_text.find(instruction) + len(instruction) if temp_text.find(instruction) != -1 else 0
             response_text = temp_text[original_instruction_end:].strip()
        else:
             response_text = response_full # Return the whole output as a last resort


    # Basic post-processing: remove potential repetition of instruction
    if response_text.startswith(instruction):
       response_text = response_text[len(instruction):].strip()

    return response_text

# --- Streamlit UI ---

# Set page config for wider layout and title
st.set_page_config(page_title="Ticketing Chatbot", layout="wide")

# CSS styling - This defines the primary button style (orange/pink gradient)
st.markdown(
    """
<style>
/* Main Button Style (Orange/Pink Gradient) */
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
    margin-top: 5px; /* Adjust slightly if needed */
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
body, .stApp * {
    font-family: 'Times New Roman', Times, serif !important;
}
/* Specific adjustments if needed (Keep these for robustness) */
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
/* Ensure messages within alerts/errors also use the font */
.stAlert *, .stException * {
    font-family: 'Times New Roman', Times, serif !important;
}
.streamlit-expanderHeader * {
     font-family: 'Times New Roman', Times, serif !important;
}
.streamlit-expanderContent * {
    font-family: 'Times New Roman', Times, serif !important;
}
</style>
    """,
    unsafe_allow_html=True,
)

# --- REMOVED THIS BLOCK ---
# The following block specifically changed the "Ask this question" button to blue.
# Removing it makes that button inherit the main orange/pink style defined above.
#
# st.markdown(
#     """
# <style>
# /* Custom CSS for the "Ask this question" button (Blue Gradient) - REMOVED */
# /* div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) { */
# /*    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */ */
# /*    color: white !important; */
# /* } */
# </style>
#     """,
#     unsafe_allow_html=True,
# )
# --- END OF REMOVED BLOCK ---


# Custom CSS for horizontal line separator
st.markdown(
    """
<style>
    .horizontal-line {
        border-top: 2px solid #e0e0e0; /* Adjust color and thickness */
        margin: 20px 0; /* Adjust spacing */
    }
</style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI Title
st.markdown("<h1 style='font-size: 43px; text-align: center;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)
st.markdown("---") # Add a visual separator

# Initialize session state for controlling disclaimer visibility
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "nlp_loaded" not in st.session_state:
    st.session_state.nlp_loaded = False


# --- Disclaimer Section ---
if not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; color: #664d03; border: 1px solid #ffecb5; font-family: 'Times New Roman', Times, serif;">
            <h2 style="font-size: 28px; color: #664d03; font-weight: bold; text-align: center;">⚠️ Disclaimer & Model Information</h2>
            <p style="font-size: 17px; line-height: 1.6;">
                Welcome! This chatbot uses a fine-tuned <b>DistilGPT-2</b> model to assist with event ticketing inquiries.
            </p>
            <p style="font-size: 17px; line-height: 1.6;">
                Due to computational constraints during fine-tuning, the model is optimized for specific intents and may provide inaccurate or generic responses for queries outside its training scope. It also uses <b>spaCy</b> for identifying entities like event names or locations in your questions.
            </p>
            <h3 style='font-size: 20px; color: #664d03;'>Optimized Intents:</h3>
            <ul style="font-size: 16px; line-height: 1.6; columns: 2; -webkit-columns: 2; -moz-columns: 2;">
                <li>Buy Ticket</li>
                <li>Sell Ticket</li>
                <li>Cancel Ticket</li>
                <li>Track Cancellation</li>
                <li>Check Cancellation Fee</li>
                <li>Transfer Ticket</li>
                <li>Upgrade Ticket</li>
                <li>Find Ticket</li>
                <li>Change Personal Details</li>
                <li>Get Refund</li>
                <li>Find Upcoming Events</li>
                <li>Customer Service</li>
                <li>Ticket Information</li>
            </ul>
            <p style="font-size: 17px; line-height: 1.6;">
                Please be patient if the model takes a moment to load initially.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("") # Add some space

    # Centered Continue button
    col1, col2, col3 = st.columns([2, 1, 2]) # Adjust ratios as needed for centering
    with col2:
        # Pre-load models when showing the disclaimer page
        if not st.session_state.nlp_loaded:
            with st.spinner("Loading language processing tools (spaCy)..."):
                 nlp = load_spacy_model()
                 st.session_state.nlp_loaded = True
                 st.session_state.nlp = nlp # Store in session state

        if not st.session_state.model_loaded:
             # The load_model_and_tokenizer function now includes spinners/messages
             model, tokenizer = load_model_and_tokenizer()
             if model and tokenizer:
                 st.session_state.model_loaded = True
                 # Store model and tokenizer in session state if needed elsewhere,
                 # but cache_resource handles re-loading efficiently.
                 st.session_state.model = model
                 st.session_state.tokenizer = tokenizer
             else:
                  st.error("Chatbot AI model failed to load. Please check logs or refresh.")
                  st.stop() # Stop execution if model fails

        # Show continue button only after models are loaded
        if st.session_state.model_loaded and st.session_state.nlp_loaded:
            if st.button("Continue to Chat", key="continue_button"):
                st.session_state.show_chat = True
                st.rerun()
        else:
             # Display a message while loading if button isn't ready
             st.info("Please wait while the chatbot initializes...")


# --- Main Chat Interface Section ---
if st.session_state.show_chat:
    st.info("Ask me about ticket cancellations, refunds, or other event inquiries!")

    # Use columns for layout: Examples on left, Chat on right (or stacked on mobile)
    col_examples, col_chat = st.columns([1, 2]) # Adjust ratio as needed

    with col_examples:
        st.markdown("#### Example Questions")
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
        selected_query = st.selectbox(
            "Choose a query:",
            ["Select an example..."] + example_queries,
            key="query_selectbox",
            label_visibility="collapsed" # Hide label as we have a header
        )
        process_query_button = st.button("Ask this question", key="query_button")

    with col_chat:
        st.markdown("#### Chat")

        # Retrieve models from session state or cache
        # Using cache_resource is generally preferred, direct retrieval shown for completeness
        # nlp = st.session_state.get('nlp', load_spacy_model())
        # model = st.session_state.get('model')
        # tokenizer = st.session_state.get('tokenizer')

        # Rely on cached resources (simpler and usually sufficient)
        nlp = load_spacy_model()
        model, tokenizer = load_model_and_tokenizer()


        if model is None or tokenizer is None or nlp is None:
            st.error("Essential components (Model/Tokenizer/NLP) failed to load. Cannot start chat.")
            st.stop()

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        last_role = None # Track last message role

        # Display chat messages
        chat_container = st.container() # Use a container for chat history
        with chat_container:
             # Determine height dynamically or set a fixed height
             # This requires custom CSS or potentially a component, difficult in pure Streamlit
             # For now, it will just grow
            for i, message in enumerate(st.session_state.chat_history):
                is_last_message = (i == len(st.session_state.chat_history) - 1)
                if message["role"] == "user" and last_role == "assistant":
                    # Add separator only if it's not the very last message overall
                    # or if there are more messages following this user message
                     if not is_last_message or len(st.session_state.chat_history) > i + 1:
                          st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

                with st.chat_message(message["role"], avatar=message["avatar"]):
                    st.markdown(message["content"], unsafe_allow_html=True)
                last_role = message["role"]


        # Process selected query from dropdown
        if process_query_button:
            if selected_query == "Select an example...":
                st.toast("⚠️ Please select a question from the dropdown first.", icon="⚠️")
            else:
                prompt_from_dropdown = selected_query
                # Capitalize first letter
                prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

                # Append user message
                st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})

                # Generate and append assistant response
                with st.spinner("Generating response..."):
                    dynamic_placeholders = extract_dynamic_placeholders(prompt_from_dropdown, nlp)
                    response_gpt = generate_response(model, tokenizer, prompt_from_dropdown)
                    full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                    # time.sleep(1) # Optional artificial delay

                st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
                # Rerun to display the new messages
                st.rerun()


        # Chat input for user's own questions
        if prompt := st.chat_input("Enter your own question here:"):
            # Capitalize first letter
            prompt = prompt[0].upper() + prompt[1:] if prompt else prompt

            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})

             # Generate and append assistant response
            with st.spinner("Generating response..."):
                 dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
                 response_gpt = generate_response(model, tokenizer, prompt)
                 full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                 # time.sleep(1) # Optional artificial delay

            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            # Rerun to display the new messages
            st.rerun()

        # Reset button at the bottom of the chat column
        if st.session_state.chat_history:
            st.markdown("---") # Separator before reset button
            if st.button("Reset Chat", key="reset_button"):
                st.session_state.chat_history = []
                st.rerun()
