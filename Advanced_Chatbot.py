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

# --- START: JavaScript for Auto-Scroll ---
# Define the JavaScript code to scroll to the bottom of the page
js_scroll_down = """
<script>
    window.scrollTo(0, document.body.scrollHeight);
</script>
"""
# --- END: JavaScript for Auto-Scroll ---

# Function to download model files from GitHub
def download_model_files(model_dir="/tmp/DistilGPT2_Model"):
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
                st.error(f"Failed to download {filename} from GitHub. Error: {e}")
                return False
    return True

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    # Try loading the transformer model first
    try:
        nlp = spacy.load("en_core_web_trf")
        return nlp
    except OSError:
        st.warning("`en_core_web_trf` model not found. Downloading...")
        try:
            spacy.cli.download("en_core_web_trf")
            nlp = spacy.load("en_core_web_trf")
            return nlp
        except Exception as e:
            st.error(f"Failed to download and load `en_core_web_trf`. Error: {e}")
            st.warning("Falling back to `en_core_web_sm`. NER accuracy might be reduced.")
            # Fallback to a smaller model if trf fails
            try:
                nlp_sm = spacy.load("en_core_web_sm")
                return nlp_sm
            except OSError:
                st.warning("`en_core_web_sm` model not found. Downloading...")
                try:
                    spacy.cli.download("en_core_web_sm")
                    nlp_sm = spacy.load("en_core_web_sm")
                    return nlp_sm
                except Exception as e_sm:
                     st.error(f"Failed to download/load even `en_core_web_sm`. NER will be unavailable. Error: {e_sm}")
                     return None # Indicate failure


# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    st.write(f"Attempting to download/load model from {model_dir}...") # Debugging info
    if not download_model_files(model_dir):
        st.error("Model download failed. Check your internet connection or GitHub URL.")
        return None, None

    try:
        # Check if model files actually exist after download attempt
        required_files_exist = all(os.path.exists(os.path.join(model_dir, fname)) for fname in MODEL_FILES)
        if not required_files_exist:
            st.error("One or more required model files are missing after download attempt.")
            # Optionally list missing files
            missing = [f for f in MODEL_FILES if not os.path.exists(os.path.join(model_dir, f))]
            st.info(f"Missing files: {', '.join(missing)}")
            return None, None

        st.write("Model files seem to exist. Loading model and tokenizer...") # Debugging info
        # Explicitly trust remote code if necessary, though usually not needed for standard models
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        st.write("Model and tokenizer loaded successfully.") # Debugging info
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        # Add more specific error handling if needed (e.g., OSError, ValueError)
        st.error("Please ensure the model files are downloaded correctly and are compatible.")
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
    "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note the space before SECTION, fix if needed
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
    # Replace dynamic placeholders last to avoid accidentally replacing parts of static values
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    dynamic_placeholders = {}
    # Default values
    dynamic_placeholders['{{EVENT}}'] = "the event"
    dynamic_placeholders['{{CITY}}'] = "your city"

    if nlp is None: # Handle case where spaCy model failed to load
        st.warning("SpaCy model not loaded. Cannot extract dynamic entities like Event or City.", icon="⚠️")
        return dynamic_placeholders

    try:
        doc = nlp(user_question)
        found_event = False
        found_city = False
        for ent in doc.ents:
            if ent.label_ == "EVENT" and not found_event:
                event_text = ent.text.title()
                dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                found_event = True # Take the first one found
            elif ent.label_ == "GPE" and not found_city: # GPE typically includes cities, countries, states
                city_text = ent.text.title()
                dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
                found_city = True # Take the first one found
            # Stop if both are found
            if found_event and found_city:
                break
    except Exception as e:
        st.warning(f"Error during NER processing: {e}. Using default placeholders.", icon="⚠️")
        # Keep default placeholders if error occurs

    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    if model is None or tokenizer is None:
         return "Sorry, the language model is not available right now."
    try:
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Format the input correctly for instruction fine-tuned models
        input_text = f"Instruction: {instruction}\nResponse:" # Added newline for clarity
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length // 2).to(device) # Truncate long instructions
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length // 2, # Generate roughly as many tokens as the input
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        # Decode the generated tokens, excluding the input prompt
        full_decoded_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Find the start of the actual response after "Response:"
        response_marker = "Response:"
        response_start_index = full_decoded_response.find(response_marker)
        if response_start_index != -1:
            response_text = full_decoded_response[response_start_index + len(response_marker):].strip()
        else:
            # Fallback if "Response:" marker is not found (shouldn't happen with the prompt format)
            response_text = full_decoded_response.replace(input_text.replace("\nResponse:", ""), "").strip() # Try removing the input text

        # Basic post-processing (can be expanded)
        response_text = response_text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
        response_text = response_text.replace(" 's", "'s").replace(" n't", "n't")

        return response_text

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, I encountered an error while generating the response."


# CSS styling
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
body, div, span, p, h1, h2, h3, h4, h5, h6, label, input, textarea, button, select, option, li, a {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Ensure chat messages use the font */
.stChatMessage * {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Specific adjustments for Streamlit elements to override defaults if necessary */
.stSelectbox label, .stTextInput label, .stTextArea label {
     font-family: 'Times New Roman', Times, serif !important;
}
.stSelectbox div[data-baseweb="select"] > div {
     font-family: 'Times New Roman', Times, serif !important;
}
.stTextInput input, .stTextArea textarea {
     font-family: 'Times New Roman', Times, serif !important;
}
div[data-testid="stExpander"] summary { /* Expander Headers */
    font-family: 'Times New Roman', Times, serif !important;
}
.streamlit-expanderContent * { /* Content inside expanders */
    font-family: 'Times New Roman', Times, serif !important;
}

/* Style for the "Ask this question" button specifically */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */
    color: white !important;
}

/* Custom CSS for horizontal line separator */
.horizontal-line {
    border-top: 1px solid #e0e0e0; /* Thinner line */
    margin: 10px 0; /* Adjust spacing */
}

/* CSS for Chat Input Shadow Effect */
div[data-testid="stChatInput"] {
    box-shadow: 0 -4px 8px rgba(0, 0, 0, 0.1); /* Shadow only on top */
    border-top: 1px solid #eee; /* Subtle top border */
    border-radius: 0; /* Remove rounding if shadow is only on top */
    padding: 10px; /* Adjust padding */
    margin: 0; /* Remove margin */
    /* position: sticky; /* Make input stick to bottom */
    /* bottom: 0; */
    /* background-color: white; /* Ensure background for sticky */
    /* z-index: 99; */ /* Ensure it's above other content */
}

/* Disclaimer Box Styling */
.disclaimer-box {
    background-color: #fff8e1; /* Light yellow */
    padding: 20px;
    border-radius: 10px;
    color: #6d4c41; /* Brownish text */
    border: 1px solid #ffecb3; /* Lighter yellow border */
    font-family: 'Times New Roman', Times, serif !important;
    margin-bottom: 20px; /* Space below disclaimer */
}
.disclaimer-box h1 {
    font-size: 30px; /* Adjusted size */
    color: #d84315; /* Deep orange title */
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
    font-family: 'Times New Roman', Times, serif !important;
}
.disclaimer-box p, .disclaimer-box ul {
    font-size: 16px;
    line-height: 1.6;
    color: #5d4037; /* Darker brownish text */
    font-family: 'Times New Roman', Times, serif !important;
}
.disclaimer-box ul {
    list-style-type: disc; /* Standard bullets */
    padding-left: 25px; /* Indent list */
}
.disclaimer-box b {
    font-weight: bold;
    color: #4e342e; /* Even darker for emphasis */
}
</style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.markdown("<h1 style='font-size: 43px; font-family: Times New Roman, Times, serif;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "nlp" not in st.session_state:
    st.session_state.nlp = None
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Example queries for dropdown
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

# --- Model Loading Logic ---
if not st.session_state.models_loaded:
    loading_placeholder = st.empty() # Create a placeholder for loading messages
    with loading_placeholder.container():
        with st.spinner("Loading models and resources... This might take a moment..."):
            try:
                # Load spaCy model first
                nlp_model = load_spacy_model()
                if nlp_model:
                    st.session_state.nlp = nlp_model
                    st.write("✔️ SpaCy NER model loaded.") # Progress update
                else:
                    st.warning("⚠️ SpaCy model could not be loaded. NER features will be limited.")
                    st.session_state.nlp = None # Ensure it's None if loading failed

                # Load DistilGPT2 model and tokenizer
                gpt_model, gpt_tokenizer = load_model_and_tokenizer()

                if gpt_model is not None and gpt_tokenizer is not None:
                    st.session_state.model = gpt_model
                    st.session_state.tokenizer = gpt_tokenizer
                    st.session_state.models_loaded = True
                    st.write("✔️ Chatbot model loaded.") # Progress update
                    time.sleep(1) # Brief pause to show final message
                    loading_placeholder.empty() # Clear the loading messages
                    st.rerun() # Rerun to proceed to the next stage (Disclaimer or Chat)
                else:
                    st.error("❌ Failed to load the chatbot model. Please check the logs above, refresh the page, and try again.")
                    # Keep the placeholder with the error message
            except Exception as e:
                st.error(f"An unexpected error occurred during model loading: {str(e)}")
                # Keep the placeholder with the error message

# --- Disclaimer Display Logic ---
elif st.session_state.models_loaded and not st.session_state.show_chat:
    st.markdown(
        """
        <div class="disclaimer-box">
            <h1>⚠️ Disclaimer</h1>
            <p>
                This <b>Chatbot</b> is designed to assist with ticketing inquiries. Due to computational limits, it's fine-tuned on specific intents and may not accurately respond to all queries.
            </p>
            <p>
                Optimized intents include:
            </p>
            <ul>
                <li>Buy, Sell, Cancel, Transfer, Upgrade, Find Tickets</li>
                <li>Change Personal Details</li>
                <li>Get Refund, Check Cancellation Fee, Track Cancellation</li>
                <li>Find Upcoming Events</li>
                <li>Customer Service, Ticket Information</li>
            </ul>
            <p>
                Queries outside these areas might not be handled correctly. If the chatbot struggles even with listed intents, please be patient and rephrase your question.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Continue button aligned to the right
    _, col_btn = st.columns([4, 1]) # Adjust ratio if needed
    with col_btn:
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun() # Rerun to display the chat interface

# --- Chat Interface Logic ---
elif st.session_state.models_loaded and st.session_state.show_chat:

    # Layout for dropdown and button at the top
    col1, col2 = st.columns([4, 1]) # Ratio for selectbox vs button
    with col1:
        selected_query = st.selectbox(
            "Choose a query from examples:",
            ["Choose your question"] + example_queries,
            key="query_selectbox",
            label_visibility="collapsed" # Hide label "Choose a query..."
        )
    with col2:
        process_query_button = st.button("Ask", key="query_button") # Shorter button label

    st.markdown("<hr style='margin-top: 5px; margin-bottom: 15px;'>", unsafe_allow_html=True) # Separator line

    # Access loaded models from session state (already checked they exist)
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    # Display chat history
    last_role = None
    for i, message in enumerate(st.session_state.chat_history):
        # Add separator line only between assistant and subsequent user message
        if message["role"] == "user" and last_role == "assistant":
             st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        last_role = message["role"]

    # --- Process Actions: Dropdown or Chat Input ---

    # Variable to check if a new response was generated in this run
    new_response_generated = False

    # 1. Process selected query from dropdown
    if process_query_button and selected_query != "Choose your question":
        prompt_from_dropdown = selected_query
        prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

        # Add user message to history and display
        st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})
        if last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt_from_dropdown, unsafe_allow_html=True)
        last_role = "user"

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Generating response..."):
                dynamic_placeholders = extract_dynamic_placeholders(prompt_from_dropdown, nlp)
                raw_response = generate_response(model, tokenizer, prompt_from_dropdown)
                full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        last_role = "assistant"
        new_response_generated = True # Mark that a response was added


    # 2. Process chat input
    if prompt := st.chat_input("Enter your own question:"):
        prompt = prompt[0].upper() + prompt[1:] if prompt else prompt # Capitalize first letter

        # Add user message to history and display
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        if last_role == "assistant":
             st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
        last_role = "user"

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Generating response..."):
                dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
                raw_response = generate_response(model, tokenizer, prompt)
                full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        last_role = "assistant"
        new_response_generated = True # Mark that a response was added

    # --- Auto-scroll and Reset Button ---

    # Inject JavaScript to scroll down AFTER new messages might have been added
    if new_response_generated:
        st.markdown(js_scroll_down, unsafe_allow_html=True)
        # We might need to rerun slightly differently if using scroll script,
        # but often just letting Streamlit finish the script run is enough.
        # If scrolling is inconsistent, a small time.sleep(0.1) before rerun *might* help,
        # or just remove the rerun here. Let's try without explicit rerun first.
        # st.rerun() # Re-running might reset the scroll effect too quickly. Avoid if possible.


    # Conditionally display reset button (aligned bottom-right maybe?)
    if st.session_state.chat_history:
        # Use columns to push the button to the right, maybe above chat input?
        # Or place it less intrusively, perhaps in a sidebar if you add one, or just below chat history.
        # For simplicity, let's place it below the main chat area for now.
        st.markdown("---") # Separator before reset
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            st.rerun() # Rerun to clear the displayed chat
