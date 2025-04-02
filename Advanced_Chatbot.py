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

    for filename in MODEL_FILES:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

        if not os.path.exists(local_path):
            try:
                response = requests.get(url, timeout=30) # Add timeout
                response.raise_for_status() # Raise an exception for bad status codes
                with open(local_path, "wb") as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from GitHub: {e}")
                return False # Stop if one file fails
    return True

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        # Try loading the transformer model first
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        st.warning("`en_core_web_trf` not found. Falling back to `en_core_web_sm`. NER might be less accurate.")
        st.info("Downloading `en_core_web_sm`...")
        try:
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"Failed to download or load spaCy model `en_core_web_sm`: {e}")
            return None
    return nlp


# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading AI Model...") # Add spinner text
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Check your internet connection or GitHub URL.")
        return None, None

    try:
        # Load model and tokenizer
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        # Optionally, try deleting the downloaded files if loading fails
        # import shutil
        # if os.path.exists(model_dir):
        #     shutil.rmtree(model_dir)
        #     st.info("Attempted to clear cached model files. Please refresh.")
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
    "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note: Potential typo here ("SUPPORT_ SECTION") - corrected below
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
# Correcting potential typo
if "{{SUPPORT_ SECTION}}" in static_placeholders:
     static_placeholders["{{SUPPORT_SECTION}}"] = static_placeholders.pop("{{SUPPORT_ SECTION}}")

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    # Replace dynamic first to avoid accidental replacement inside static values
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    # Then replace static placeholders
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    dynamic_placeholders = {}
    # Default values
    dynamic_placeholders['{{EVENT}}'] = "the event" # More generic default
    dynamic_placeholders['{{CITY}}'] = "the city"   # More generic default

    if nlp is None:
         st.warning("SpaCy model not loaded. Cannot extract EVENT or CITY details.")
         return dynamic_placeholders # Return defaults if NLP model failed to load

    try:
        doc = nlp(user_question)
        event_found = False
        city_found = False
        for ent in doc.ents:
            # Prioritize ORG or EVENT for event name, GPE for city
            if ent.label_ in ["EVENT", "ORG"] and not event_found: # Check if already found
                event_text = ent.text.title()
                dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                event_found = True
            elif ent.label_ == "GPE" and not city_found: # Check if already found
                city_text = ent.text.title()
                dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
                city_found = True
            # Break if both found to avoid overwriting with less specific entities
            if event_found and city_found:
                break
    except Exception as e:
        st.error(f"Error during Named Entity Recognition: {e}")
        # Keep default placeholders if NER fails

    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = f"Instruction: {instruction} Response:"
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device) # Add truncation
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
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Find the response part more robustly
        response_marker = "Response:"
        response_start_index = response.find(response_marker)
        if response_start_index != -1:
            response_text = response[response_start_index + len(response_marker):].strip()
            # Sometimes the model might output the instruction again, remove it if present
            if response_text.startswith(instruction):
                 response_text = response_text[len(instruction):].strip()
            return response_text
        else:
            # Fallback if "Response:" marker is missing (less likely with this prompt format)
            return response.strip() # Or return a default error message

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, I encountered an error while generating the response."


# --- Streamlit UI ---

# CSS styling (kept as is)
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
body, div, span, p, h1, h2, h3, h4, h5, h6, li, input, textarea, button, select, option {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Specific adjustments for Streamlit elements */
.stSelectbox div[data-baseweb="select"] > div { /* Target selectbox text */
    font-family: 'Times New Roman', Times, serif !important;
}
.stTextInput input { /* Target text input */
    font-family: 'Times New Roman', Times, serif !important;
}
.stTextArea textarea { /* Target text area */
    font-family: 'Times New Roman', Times, serif !important;
}
.stChatMessage { /* Target chat messages */
     font-family: 'Times New Roman', Times, serif !important;
}
.stAlert { /* Target alerts like st.error, st.warning */
    font-family: 'Times New Roman', Times, serif !important;
}
.streamlit-expanderHeader { /* Expander header */
    font-family: 'Times New Roman', Times, serif !important;
}
.streamlit-expanderContent { /* Expander content */
    font-family: 'Times New Roman', Times, serif !important;
}
p, li { /* Ensure paragraphs and list items inherit */
   font-family: inherit !important;
}
h1, h2, h3, h4, h5, h6 { /* Ensure headings inherit */
   font-family: inherit !important;
}

</style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for the "Ask this question" button (kept as is)
st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */
    color: white !important;
}
</style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for horizontal line separator (kept as is)
st.markdown(
    """
<style>
    .horizontal-line {
        border-top: 2px solid #e0e0e0; /* Adjust color and thickness as needed */
        margin: 15px 0; /* Adjust spacing above and below the line */
    }
</style>
    """,
    unsafe_allow_html=True,
)


st.markdown("<h1 style='font-size: 43px; text-align: center;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)


# Initialize session state for controlling disclaimer visibility
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state: # Initialize chat history early
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
    "Can I sell my ticket?"
]

# Display Disclaimer and Continue button if chat hasn't started
if not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb; font-family: 'Times New Roman', Times, serif !important;">
            <h1 style="font-size: 30px; color: #721c24; font-weight: bold; text-align: center; font-family: inherit !important;">⚠️ Disclaimer</h1>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24; font-family: inherit !important;">
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents and may not be able to respond accurately to all types of queries.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24; font-family: inherit !important;">
                The chatbot is optimized to handle the following intents:
            </p>
            <ul style="font-size: 16px; line-height: 1.6; color: #721c24; margin-left: 20px; font-family: inherit !important;">
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
            <p style="font-size: 16px; line-height: 1.6; color: #721c24; font-family: inherit !important;">
                Please note that this chatbot may not be able to assist with queries outside of these predefined intents. For complex issues, please contact our human support team via the <b>Customer Service</b> section.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- MODIFICATION START: Use columns to align button ---
    st.write("") # Add a little vertical space before the button
    col1, col2, col3 = st.columns([2, 3, 1.2]) # Create 3 columns: empty spacer, empty spacer, button column

    with col3: # Use the rightmost column for the button
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun() # Rerun the script immediately to hide disclaimer and show chat
    # --- MODIFICATION END ---

# Show chat interface only after clicking Continue
if st.session_state.show_chat:

    # Load models only when needed (after clicking continue)
    nlp = load_spacy_model()
    model, tokenizer = load_model_and_tokenizer()

    # Check if models loaded successfully
    if model is None or tokenizer is None or nlp is None:
        st.error("Critical components failed to load. Chatbot cannot function. Please check logs or try refreshing.")
        st.stop() # Stop execution if models aren't loaded


    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # --- Layout for Dropdown and "Ask this question" button ---
    col_select, col_button = st.columns([4, 1]) # Give more space to selectbox

    with col_select:
        selected_query = st.selectbox(
            "Choose a query from examples:",
            ["Choose your question"] + example_queries,
            key="query_selectbox",
            label_visibility="collapsed" # Hide the label itself
        )
    with col_button:
        process_query_button = st.button("Ask", key="query_button", help="Ask the selected example question") # Shorter text, add tooltip

    # --- Chat History Display ---
    last_role = None # Track last message role for separator logic
    for i, message in enumerate(st.session_state.chat_history):
        # Add separator only between user and subsequent assistant message
        if message["role"] == "user" and i > 0 and st.session_state.chat_history[i-1]["role"] == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        last_role = message["role"] # Update last role after displaying


    # --- Process selected query from dropdown ---
    if process_query_button:
        if selected_query == "Choose your question":
            st.toast("⚠️ Please select a question from the dropdown first.", icon="⚠️") # Use toast for non-blocking message
        elif selected_query:
            prompt_from_dropdown = selected_query
            # Capitalize first letter (optional, but good practice)
            prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

            # Add user message to history and display
            st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})
            if last_role == "assistant": # Check if separator needed before displaying
                 st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt_from_dropdown, unsafe_allow_html=True)
            last_role = "user"

            # Generate and display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                generating_response_text = "Generating response..."
                with st.spinner(generating_response_text):
                    dynamic_placeholders = extract_dynamic_placeholders(prompt_from_dropdown, nlp)
                    raw_response = generate_response(model, tokenizer, prompt_from_dropdown)
                    full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)
                    # time.sleep(1) # Optional delay - usually not needed

                message_placeholder.markdown(full_response, unsafe_allow_html=True)

            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            last_role = "assistant"

            # Clear the selectbox after processing to avoid resubmission on rerun
            st.session_state.query_selectbox = "Choose your question"
            st.rerun() # Rerun to reflect the cleared selectbox and updated history


    # --- Input box at the bottom ---
    if prompt := st.chat_input("Enter your own question:"):
        # Capitalize first letter (optional)
        prompt = prompt[0].upper() + prompt[1:] if prompt else prompt

        # Add user message to history and display
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        if last_role == "assistant": # Check if separator needed before displaying
             st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
        last_role = "user"

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            generating_response_text = "Generating response..."
            with st.spinner(generating_response_text):
                dynamic_placeholders = extract_dynamic_placeholders(prompt, nlp)
                raw_response = generate_response(model, tokenizer, prompt)
                full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)
                # time.sleep(1) # Optional delay

            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        last_role = "assistant"
        st.rerun() # Rerun to show the latest message immediately


    # --- Conditionally display reset button at the bottom ---
    if st.session_state.chat_history:
        st.markdown("---") # Add a visual separator before the reset button
        # Use columns to push the reset button to the right
        reset_col1, reset_col2, reset_col3 = st.columns([4, 3, 1.2])
        with reset_col3:
            if st.button("Reset Chat", key="reset_button", help="Clear the conversation history"):
                st.session_state.chat_history = []
                last_role = None
                st.rerun() # Rerun to clear the chat display
