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
    download_successful = True
    for filename in MODEL_FILES:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

        if not os.path.exists(local_path):
            try:
                response = requests.get(url, timeout=30) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                with open(local_path, "wb") as f:
                    f.write(response.content)
                # st.success(f"Successfully downloaded {filename}") # Optional: success message
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from GitHub. Error: {e}")
                download_successful = False
                break # Stop trying if one file fails
    return download_successful

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        # Try loading the transformer model first
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        st.warning("`en_core_web_trf` not found. Downloading... This may take a while.")
        try:
            spacy.cli.download("en_core_web_trf")
            nlp = spacy.load("en_core_web_trf")
            st.success("`en_core_web_trf` downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download `en_core_web_trf`. Falling back to `en_core_web_sm`. Error: {e}")
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                st.warning("`en_core_web_sm` not found. Downloading...")
                try:
                    spacy.cli.download("en_core_web_sm")
                    nlp = spacy.load("en_core_web_sm")
                    st.success("`en_core_web_sm` downloaded successfully.")
                except Exception as e_sm:
                     st.error(f"Failed to download `en_core_web_sm`. NER features will be limited. Error: {e_sm}")
                     return None # Return None if both fail
    return nlp


# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading AI Model...") # Added spinner message
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot proceed without the model files.")
        return None, None

    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        st.error("Please ensure all model files were downloaded correctly and are not corrupted.")
        # Attempt to clean up potentially corrupted files so download can be retried
        if os.path.exists(model_dir):
            import shutil
            try:
                shutil.rmtree(model_dir)
                st.info("Removed potentially corrupted model cache. Please refresh the page to retry download.")
            except Exception as rm_e:
                st.error(f"Could not remove model cache directory: {rm_e}")
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
    "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note: potential typo here "SUPPORT_ SECTION" vs "SUPPORT_SECTION"
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
    # Prioritize dynamic placeholders if there's overlap (e.g., {{EVENT}})
    # First replace static ones
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    # Then replace dynamic ones, potentially overwriting defaults
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    dynamic_placeholders = {
        '{{EVENT}}': "the event", # Default value
        '{{CITY}}': "the city"    # Default value
    }
    if nlp and user_question: # Check if nlp model loaded and question is not empty
        try:
            doc = nlp(user_question)
            event_found = False
            city_found = False
            for ent in doc.ents:
                if ent.label_ == "EVENT" and not event_found:
                    event_text = ent.text.title()
                    dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                    event_found = True
                elif ent.label_ == "GPE" and not city_found: # GPE (Geopolitical Entity) often catches cities/locations
                    city_text = ent.text.title()
                    dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
                    city_found = True
                # Stop if both are found
                if event_found and city_found:
                    break
        except Exception as e:
            st.warning(f"NER processing failed: {e}. Using default placeholders.")
            # Ensure defaults are still set if error occurs mid-processing
            if '{{EVENT}}' not in dynamic_placeholders: dynamic_placeholders['{{EVENT}}'] = "the event"
            if '{{CITY}}' not in dynamic_placeholders: dynamic_placeholders['{{CITY}}'] = "the city"

    return dynamic_placeholders


# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    if not model or not tokenizer:
        return "Error: Model or tokenizer not loaded."

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure instruction is not empty
    if not instruction or not instruction.strip():
        return "Please provide a question."

    input_text = f"Instruction: {instruction} Response:"
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length).to(device) # Added truncation
        # inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

        # Check if inputs are empty after tokenization (edge case)
        if inputs["input_ids"].shape[1] == 0:
             return "I couldn't understand the input. Please try rephrasing."

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=min(max_length, tokenizer.model_max_length + inputs["input_ids"].shape[1]), # Adjust max_length based on input
                # max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id # Explicitly set eos_token_id for generation
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Find the response part more robustly
        response_marker = "Response:"
        response_start_index = response.find(response_marker)
        if response_start_index != -1:
            response_text = response[response_start_index + len(response_marker):].strip()
        else:
            # Fallback if "Response:" marker is not found (e.g., model didn't follow format)
            # Attempt to remove the instruction part if it's present at the start
            instruction_marker = "Instruction:"
            instruction_end_index = response.find(instruction_marker)
            if instruction_end_index != -1:
                 # Find where the instruction likely ends before the response starts
                 potential_response_start = response.find(instruction, instruction_end_index) + len(instruction)
                 response_text = response[potential_response_start:].strip()
                 # Basic cleanup if needed
                 if response_text.startswith(":"): response_text = response_text[1:].strip()

            else:
                 response_text = response.strip() # Use the whole output as response


        # Post-processing: Remove potential artifacts like repeated instructions if necessary
        if response_text.startswith(instruction):
             response_text = response_text[len(instruction):].strip()
        if response_text.startswith(":"):
             response_text = response_text[1:].strip()

        # Ensure response is not empty
        if not response_text:
            return "I received an empty response from the model. Could you please try asking differently?"

        return response_text

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, I encountered an error while generating the response."

# --- CSS Styling ---
st.markdown(
    """
<style>
/* === General Styles === */
* {
    font-family: 'Times New Roman', Times, serif !important;
}

/* === Button Styles === */
.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71); /* Stylish gradient */
    color: white !important; /* Ensure text is white */
    border: none;
    border-radius: 25px; /* Rounded corners */
    padding: 10px 20px; /* Padding */
    font-size: 1.1em; /* Adjusted font size slightly */
    font-weight: bold; /* Bold text */
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease; /* Smooth transitions */
    display: inline-flex; /* Helps with alignment */
    align-items: center;
    justify-content: center;
    margin-top: 5px; /* Alignment */
    width: auto; /* Fit content width */
    min-width: 100px; /* Optional: ensure a minimum width */
    font-family: 'Times New Roman', Times, serif !important; /* Specific font */
}
.stButton>button:hover {
    transform: scale(1.05); /* Slightly larger on hover */
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); /* Shadow on hover */
    color: white !important; /* Ensure text stays white on hover */
}
.stButton>button:active {
    transform: scale(0.98); /* Slightly smaller when clicked */
}

/* Style for the "Ask this question" button specifically */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */
    color: white !important;
}

/* === Chat Input Styling (NEW) === */
/* Style the chat input text area */
textarea[data-testid="stChatInput"] {
    background-color: #f0f2f6 !important; /* Light gray background */
    border: 1px solid #cccccc !important; /* Gray border */
    color: #31333F !important; /* Darker text color for input */
    border-radius: 0.5rem !important; /* Match Streamlit's default radius */
    padding: 0.75rem 1rem !important; /* Adjust padding if needed */
    font-family: 'Times New Roman', Times, serif !important; /* Specific font */
}

/* Style the placeholder text within the chat input */
textarea[data-testid="stChatInput"]::placeholder {
    color: #6f747e !important; /* Darker gray for placeholder */
    opacity: 1 !important; /* Ensure placeholder is fully visible */
    font-family: 'Times New Roman', Times, serif !important; /* Specific font */
}

/* Optional: Style the outer container of the chat input if needed */
/*
div[data-testid="stChatInputContainer"] {
   border-top: 1px solid #e0e0e0; /* Example: add a top border */
/* padding: 10px 0; /* Add padding */
/*}
*/

/* === Other Element Styling === */
/* Ensure specific Streamlit elements also use the font */
.stSelectbox > div > div > div > div {
    font-family: 'Times New Roman', Times, serif !important;
}
.stTextInput > div > div > input {
    font-family: 'Times New Roman', Times, serif !important;
}
.stTextArea > div > div > textarea { /* This might conflict with chat input, ensure specificity */
    font-family: 'Times New Roman', Times, serif !important;
}
.stChatMessage {
    font-family: 'Times New Roman', Times, serif !important;
}
.st-emotion-cache-r421ms { /* Example class for st.error, st.warning - Inspect to confirm */
    font-family: 'Times New Roman', Times, serif !important;
}
.streamlit-expanderContent { /* For text inside expanders */
    font-family: 'Times New Roman', Times, serif !important;
}

/* Custom CSS for horizontal line separator */
.horizontal-line {
    border-top: 2px solid #e0e0e0; /* Gray line */
    margin: 15px 0; /* Spacing */
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
if "last_role" not in st.session_state:
    st.session_state.last_role = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "nlp_loaded" not in st.session_state:
     st.session_state.nlp_loaded = False


# --- Load Models ---
# Load spaCy model only once
if not st.session_state.nlp_loaded:
    with st.spinner("Loading NLP model for entity recognition..."):
        nlp = load_spacy_model()
        if nlp:
            st.session_state.nlp_loaded = True
        # No need for else, error/warning handled in load_spacy_model

# Load GPT model and tokenizer only once
if not st.session_state.model_loaded:
    # Use a placeholder while loading
    loading_placeholder = st.empty()
    loading_placeholder.info("Initializing the AI model. This might take a moment...") # More user-friendly
    model, tokenizer = load_model_and_tokenizer()
    if model and tokenizer:
        st.session_state.model_loaded = True
        loading_placeholder.empty() # Remove the loading message
    else:
        # Error displayed in load_model_and_tokenizer
        loading_placeholder.error("Failed to initialize the AI model. The chatbot may not function correctly.")
        st.stop() # Stop execution if model loading fails critically

# --- Display Disclaimer or Chat ---

# Example queries
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

# Display Disclaimer and Continue button if chat hasn't started
if not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb;">
            <h1 style='font-size: 36px; color: #721c24; font-weight: bold; text-align: center; font-family: "Times New Roman", Times, serif !important;'>⚠️ Disclaimer</h1>
            <p style='font-size: 16px; line-height: 1.6; color: #721c24; font-family: "Times New Roman", Times, serif !important;'>
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents and may not be able to respond accurately to all types of queries.
            </p>
            <p style='font-size: 16px; line-height: 1.6; color: #721c24; font-family: "Times New Roman", Times, serif !important;'>
                The chatbot is optimized to handle the following intents:
            </p>
            <ul style='font-size: 16px; line-height: 1.6; color: #721c24; margin-left: 20px; font-family: "Times New Roman", Times, serif !important;'>
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
            <p style='font-size: 16px; line-height: 1.6; color: #721c24; font-family: "Times New Roman", Times, serif !important;'>
                Please note that this chatbot may not be able to assist with queries outside of these predefined intents. Responses for other topics might be generic or inaccurate.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Continue button aligned to the right
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun()

# Show chat interface only after clicking Continue
elif st.session_state.show_chat and st.session_state.model_loaded: # Ensure model is loaded before showing chat
    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # --- Dropdown and Button section ---
    st.markdown("---") # Visual separator
    col_select, col_button = st.columns([4, 1]) # Adjust ratio as needed
    with col_select:
         selected_query = st.selectbox(
            "Choose a query from examples:",
            ["Choose your question"] + example_queries,
            index=0, # Default to "Choose your question"
            key="query_selectbox",
            label_visibility="collapsed"
        )
    with col_button:
        process_query_button = st.button("Ask this", key="query_button") # Shorter button text
    st.markdown("---") # Visual separator


    # --- Display Chat History ---
    for message in st.session_state.chat_history:
        # Add separator if the last message was from the assistant
        if message["role"] == "user" and st.session_state.get("last_role") == "assistant":
             st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        st.session_state.last_role = message["role"] # Update last role after displaying

    # --- Process Dropdown Query ---
    if process_query_button:
        if selected_query == "Choose your question":
            st.toast("⚠️ Please select a question from the dropdown first.", icon="⚠️")
        else:
            prompt_to_process = selected_query
            # Add user message to history and display
            if st.session_state.get("last_role") == "assistant":
                 st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
            user_message = {"role": "user", "content": prompt_to_process, "avatar": "👤"}
            st.session_state.chat_history.append(user_message)
            with st.chat_message(user_message["role"], avatar=user_message["avatar"]):
                st.markdown(user_message["content"], unsafe_allow_html=True)
            st.session_state.last_role = "user"

            # Generate and display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                with st.spinner("Generating response..."):
                    # Use the loaded nlp model from state if available
                    current_nlp = nlp if 'nlp' in locals() and st.session_state.nlp_loaded else None
                    dynamic_placeholders = extract_dynamic_placeholders(prompt_to_process, current_nlp)
                    raw_response = generate_response(model, tokenizer, prompt_to_process)
                    full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)
                    # time.sleep(1) # Optional delay for realism

                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                assistant_message = {"role": "assistant", "content": full_response, "avatar": "🤖"}
                st.session_state.chat_history.append(assistant_message)
                st.session_state.last_role = "assistant"

            # Reset selectbox after processing
            st.session_state.query_selectbox = "Choose your question"
            # st.rerun() # Rerun might be too disruptive here, let user see response first

    # --- Process User Input from Chat Input ---
    if prompt := st.chat_input("Enter your own question here..."): # Changed placeholder text slightly
        prompt = prompt.strip()
        if not prompt:
            st.toast("⚠️ Please enter a question.", icon="⚠️")
        else:
            # Capitalize first letter (optional)
            prompt_display = prompt[0].upper() + prompt[1:] if prompt else prompt

            # Add user message to history and display
            if st.session_state.get("last_role") == "assistant":
                 st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
            user_message = {"role": "user", "content": prompt_display, "avatar": "👤"}
            st.session_state.chat_history.append(user_message)
            with st.chat_message(user_message["role"], avatar=user_message["avatar"]):
                st.markdown(user_message["content"], unsafe_allow_html=True)
            st.session_state.last_role = "user"

            # Generate and display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                with st.spinner("Generating response..."):
                     # Use the loaded nlp model from state if available
                    current_nlp = nlp if 'nlp' in locals() and st.session_state.nlp_loaded else None
                    dynamic_placeholders = extract_dynamic_placeholders(prompt, current_nlp) # Use original prompt for NER
                    raw_response = generate_response(model, tokenizer, prompt) # Use original prompt for generation
                    full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)
                    # time.sleep(1) # Optional delay

                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                assistant_message = {"role": "assistant", "content": full_response, "avatar": "🤖"}
                st.session_state.chat_history.append(assistant_message)
                st.session_state.last_role = "assistant"
            # No need to rerun, chat input clears automatically

    # --- Reset Button ---
    if st.session_state.chat_history:
        st.markdown("---") # Separator before reset button
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            st.session_state.last_role = None
            st.rerun()

elif not st.session_state.model_loaded:
     # This state should ideally be handled by the loading logic above,
     # but added as a fallback message if the chat interface is somehow
     # reached before the model is ready.
     st.warning("The AI model is still loading or failed to load. Please wait or refresh the page.")
