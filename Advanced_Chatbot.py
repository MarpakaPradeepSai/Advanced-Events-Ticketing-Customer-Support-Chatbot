import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time

# --- Constants and Setup ---
# (Keep this section exactly as in your original code)
GITHUB_MODEL_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"
MODEL_FILES = [
    "config.json", "generation_config.json", "merges.txt",
    "model.safetensors", "special_tokens_map.json",
    "tokenizer_config.json", "vocab.json"
]
MODEL_DIR = "/tmp/DistilGPT2_Model"

# --- Functions ---
# (Keep all functions exactly as in your original code:
# download_model_files, load_spacy_model, load_model_and_tokenizer,
# static_placeholders, replace_placeholders, extract_dynamic_placeholders,
# generate_response)

# Function to download model files from GitHub
def download_model_files(model_dir=MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    for filename in MODEL_FILES:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(local_path):
            try:
                response = requests.get(url, timeout=30) # Add timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                with open(local_path, "wb") as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from GitHub: {e}")
                return False
    return True

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        # Try loading the specific model, download if needed
        spacy.cli.download("en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")
        return nlp
    except Exception as e:
        # Fallback logic (keep as is)
        st.error(f"Failed to load spaCy model 'en_core_web_trf': {e}")
        st.info("Attempting to load 'en_core_web_sm' as fallback...")
        try:
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            st.warning("Loaded smaller spaCy model 'en_core_web_sm'. NER might be less accurate.")
            return nlp
        except Exception as e2:
            st.error(f"Failed to load fallback spaCy model 'en_core_web_sm': {e2}")
            return None


# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading Language Model...") # Changed spinner text slightly
def load_model_and_tokenizer():
    if not download_model_files(MODEL_DIR):
        st.error("Model download failed. Check your internet connection or GitHub URL.")
        return None, None
    try:
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {MODEL_DIR}: {e}")
        return None, None

# Define static placeholders (keep as is)
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
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>", "{{SEND_BUTTON}}": "<b>Send</b>", "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note space in original key
    "{{SUPPORT_CONTACT_LINK}}": "www.support-team.com", "{{SUPPORT_SECTION}}": "<b>Support</b>", "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{TICKET_AVAILABILITY_TAB}}": "<b>Ticket Availability</b>", "{{TICKET_DETAILS}}": "<b>Ticket Details</b>", "{{TICKET_INFORMATION}}": "<b>Ticket Information</b>",
    "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>", "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>", "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>", "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>", "{{TICKET_TRANSFER_TAB}}": "<b>Ticket Transfer</b>",
    "{{TICKETING_PAGE}}": "<b>Ticketing</b>", "{{TICKETS_TAB}}": "<b>Tickets</b>", "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>", "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>", "{{UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>", "{{UPGRADE_TICKET_INFORMATION}}": "<b>Ticket Upgradation</b>", "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>", "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>", "{{WEBSITE_URL}}": "www.events-ticketing.com"
}

# Function to replace placeholders (keep as is)
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    response = str(response)
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy (keep as is)
def extract_dynamic_placeholders(user_question, nlp):
    dynamic_placeholders = {}
    if nlp:
        try:
            doc = nlp(user_question)
            for ent in doc.ents:
                if ent.label_ == "EVENT":
                    event_text = ent.text.title()
                    dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                elif ent.label_ in ["GPE", "LOC"]:
                    city_text = ent.text.title()
                    dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
        except Exception as e:
            st.warning(f"NER processing failed: {e}")
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "the event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "your city"
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2 (keep as is, maybe ensure error handling)
def generate_response(model, tokenizer, instruction, max_length=256):
    if not model or not tokenizer:
        return "Error: Model or tokenizer not loaded."
    try:
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_text = f"Instruction: {instruction} Response:"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length, # Use max_new_tokens
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, I encountered an error while generating the response."


# --- CSS Styling ---
# (Keep your original CSS block, just ADD the regenerate button style inside)
st.markdown(
    """
<style>
/* Keep ALL your existing styles here */
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
.stSelectbox > div > div > div > div { font-family: 'Times New Roman', Times, serif !important; }
.stTextInput > div > div > input { font-family: 'Times New Roman', Times, serif !important; }
.stTextArea > div > div > textarea { font-family: 'Times New Roman', Times, serif !important; }
.stChatMessage { font-family: 'Times New Roman', Times, serif !important; }
.st-emotion-cache-r421ms { font-family: 'Times New Roman', Times, serif !important; }
.streamlit-expanderContent { font-family: 'Times New Roman', Times, serif !important; }

/* Custom CSS for the "Ask this question" button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6) !important; /* Different gradient */
    color: white !important;
}

/* Custom CSS for horizontal line separator */
.horizontal-line {
    border-top: 2px solid #e0e0e0; /* Adjust color and thickness as needed */
    margin: 15px 0; /* Adjust spacing above and below the line */
}

/* CSS for Chat Input Shadow Effect */
div[data-testid="stChatInput"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}

/* --- ADD THIS STYLE FOR THE REGENERATE BUTTON --- */
div[data-testid="stChatMessage"] div[data-testid="stButton"] button {
    background-color: #f0f2f6 !important; /* Light gray background */
    color: #333 !important; /* Darker text */
    border: 1px solid #ccc !important; /* Subtle border */
    border-radius: 50% !important; /* Make it round */
    padding: 0px !important; /* Adjusted padding for perfect circle */
    font-size: 1.0em !important; /* Adjust icon size if needed */
    font-weight: normal !important; /* Normal weight */
    width: 28px !important;
    height: 28px !important;
    margin-left: 8px !important; /* Space it slightly from the text */
    margin-top: 0px !important; /* Align vertically */
    line-height: 1; /* Ensure icon is centered vertically */
    background: #f0f2f6 !important;
}
div[data-testid="stChatMessage"] div[data-testid="stButton"] button:hover {
    background-color: #e0e2e6 !important; /* Slightly darker on hover */
    color: #000 !important;
    transform: scale(1.1) !important; /* Slightly larger */
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2) !important;
}
div[data-testid="stChatMessage"] div[data-testid="stButton"] button:active {
    transform: scale(1.0) !important; /* Smaller click effect */
}
/* --- END OF ADDED STYLE --- */

</style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit UI ---
st.markdown("<h1 style='font-size: 43px;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state for controlling disclaimer visibility and model loading status
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
# --- ADDED STATE ---
# Initialize chat history and regeneration tracker in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "regenerate_index" not in st.session_state: # To track which message to regenerate
    st.session_state.regenerate_index = None
# --- END ADDED STATE ---


# Example queries for dropdown (keep as is)
example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?",
    "How can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?",
    "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation status?",
    "How can I sell my ticket?"
]

# --- Model Loading Logic ---
# (Keep this section exactly as in your original code)
if not st.session_state.models_loaded:
    with st.spinner("Loading models and resources... Please wait..."):
        try:
            # Initialize spaCy model for NER
            nlp = load_spacy_model()

            # Load DistilGPT2 model and tokenizer
            model, tokenizer = load_model_and_tokenizer()

            if model is not None and tokenizer is not None and nlp is not None: # Ensure nlp also loaded
                st.session_state.models_loaded = True
                st.session_state.nlp = nlp
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                # Don't rerun here yet, let it flow to disclaimer
            elif model is None or tokenizer is None:
                 st.error("Failed to load the language model. Please refresh the page and try again.")
            elif nlp is None:
                 st.error("Failed to load the NER model. Placeholder replacement might be affected.")
                 # Decide if you want to proceed without NER or stop
                 # To proceed without NER:
                 st.session_state.models_loaded = True # Still mark as loaded if LM is okay
                 st.session_state.nlp = None
                 st.session_state.model = model
                 st.session_state.tokenizer = tokenizer

        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
    # If models loaded successfully in the previous step, rerun to show disclaimer/chat
    if st.session_state.models_loaded:
        st.rerun()


# --- Disclaimer Logic ---
# (Keep this section exactly as in your original code)
if st.session_state.models_loaded and not st.session_state.show_chat:
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
                <li>Cancel Ticket</li> <li>Buy Ticket</li> <li>Sell Ticket</li> <li>Transfer Ticket</li>
                <li>Upgrade Ticket</li> <li>Find Ticket</li> <li>Change Personal Details on Ticket</li>
                <li>Get Refund</li> <li>Find Upcoming Events</li> <li>Customer Service</li>
                <li>Check Cancellation Fee</li> <li>Track Cancellation</li> <li>Ticket Information</li>
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


# --- Chat Interface Logic ---
if st.session_state.models_loaded and st.session_state.show_chat:
    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # --- ADDED REGENERATION HANDLING BLOCK ---
    # Access models needed for regeneration
    nlp = st.session_state.get('nlp')
    model = st.session_state.get('model')
    tokenizer = st.session_state.get('tokenizer')

    # Check if a regeneration was requested in the *previous* run
    idx_to_regenerate = st.session_state.get('regenerate_index', None)
    if idx_to_regenerate is not None:
        # Important: Reset the trigger *immediately* to avoid re-triggering loops
        st.session_state.regenerate_index = None

        # Check if index is valid and refers to an assistant message preceded by a user message
        if 0 < idx_to_regenerate < len(st.session_state.chat_history) and \
           st.session_state.chat_history[idx_to_regenerate - 1]["role"] == "user":

            original_prompt = st.session_state.chat_history[idx_to_regenerate - 1]["content"]

            # Show spinner while regenerating
            with st.spinner("🔄 Regenerating response..."):
                # Ensure models are available before proceeding
                if model and tokenizer:
                    dynamic_placeholders = extract_dynamic_placeholders(original_prompt, nlp) # nlp might be None, handled in function
                    new_response_gpt = generate_response(model, tokenizer, original_prompt)
                    new_full_response = replace_placeholders(new_response_gpt, dynamic_placeholders, static_placeholders)

                    # Update the specific message in the history
                    st.session_state.chat_history[idx_to_regenerate]["content"] = new_full_response
                    st.rerun() # Rerun Streamlit to reflect the change in the chat display
                else:
                    st.error("Cannot regenerate response: Model not available.")
        else:
            # Optional: Add a warning if regeneration failed due to index issues
            st.warning(f"Could not regenerate message at index {idx_to_regenerate}.")
    # --- END OF REGENERATION HANDLING BLOCK ---


    # Dropdown and Button section at the TOP
    # (Keep this section exactly as in your original code)
    selected_query = st.selectbox(
        "Choose a query from examples:",
        ["Choose your question"] + example_queries,
        key="query_selectbox",
        label_visibility="collapsed"
    )
    process_query_button = st.button("Ask this question", key="query_button")

    # Access loaded models from session state for regular processing

    last_role = None # Track last message role

    # Display chat messages from history
    for idx, message in enumerate(st.session_state.chat_history): # Add enumerate to get index
        if message["role"] == "user" and last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)

            # --- ADDED REGENERATE BUTTON ---
            # Add regenerate button ONLY for assistant messages that follow a user message
            if message["role"] == "assistant" and idx > 0 and st.session_state.chat_history[idx - 1]["role"] == "user":
                button_key = f"regenerate_{idx}"
                # Add the button. When clicked, it sets the session state and reruns.
                if st.button("🔄", key=button_key, help="Regenerate this response"):
                    st.session_state.regenerate_index = idx # Store the index to regenerate
                    st.rerun() # Trigger rerun to handle regeneration at the top
            # --- END ADDED REGENERATE BUTTON ---

        last_role = message["role"]

    # Process selected query from dropdown
    # (Keep this section exactly as in your original code)
    if process_query_button:
        if selected_query == "Choose your question":
            st.error("⚠️ Please select your question from the dropdown.") # Changed to error/toast
        elif selected_query:
            prompt_from_dropdown = selected_query
            prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

            st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})

            # Retrieve models for generation
            nlp = st.session_state.get('nlp')
            model = st.session_state.get('model')
            tokenizer = st.session_state.get('tokenizer')

            if model and tokenizer: # Check if models are loaded before generating
                with st.chat_message("assistant", avatar="🤖"): # Show temporary message
                    message_placeholder = st.empty()
                    generating_response_text = "Generating response..."
                    with st.spinner(generating_response_text):
                        dynamic_placeholders = extract_dynamic_placeholders(prompt_from_dropdown, nlp) # nlp might be None
                        response_gpt = generate_response(model, tokenizer, prompt_from_dropdown)
                        full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
                    st.rerun() # Rerun to display the full history including new message
            else:
                 st.error("Cannot generate response: Model not loaded.")


    # Input box at the bottom
    if prompt := st.chat_input("Enter your own question:"):
        prompt = prompt.strip() # Strip whitespace first
        if not prompt:
            st.toast("⚠️ Please enter a question.", icon="✏️")
        else:
            prompt = prompt[0].upper() + prompt[1:]
            st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
            nlp = st.session_state.get('nlp')
            model = st.session_state.get('model')
            tokenizer = st.session_state.get('tokenizer')

            if model and tokenizer:
                with st.chat_message("assistant", avatar="🤖"):
                    message_placeholder = st.empty()
                    generating_response_text = "Generating response..."
                    with st.spinner(generating_response_text):
                        dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
                        response_gpt = generate_response(model, tokenizer, prompt)
                        full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
                    st.rerun()
            else:
                 st.error("Cannot generate response: Model not loaded.")


    # Conditionally display reset button
    if st.session_state.chat_history:
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            st.session_state.regenerate_index = None
            last_role = None
            st.rerun()
