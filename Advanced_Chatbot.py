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
    missing_files = False
    for filename in MODEL_FILES:
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

        if not os.path.exists(local_path):
            try:
                response = requests.get(url, timeout=30) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                with open(local_path, "wb") as f:
                    f.write(response.content)
                # st.info(f"Downloaded {filename}") # Optional: Progress info
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from GitHub. Error: {e}")
                missing_files = True
                break # Stop trying if one fails

    # Verify all files are present after download attempt
    if not missing_files:
        for filename in MODEL_FILES:
            local_path = os.path.join(model_dir, filename)
            if not os.path.exists(local_path):
                st.error(f"Model file {filename} is still missing after download attempt.")
                missing_files = True
                break
    return not missing_files # Return True if all files exist, False otherwise

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    # Try loading the transformer model first
    try:
        nlp = spacy.load("en_core_web_trf")
        st.success("Loaded spaCy 'en_core_web_trf' model.")
        return nlp
    except OSError:
        st.warning("Could not load 'en_core_web_trf'. Downloading...")
        try:
            spacy.cli.download("en_core_web_trf")
            nlp = spacy.load("en_core_web_trf")
            st.success("Successfully downloaded and loaded 'en_core_web_trf'.")
            return nlp
        except Exception as e_trf:
            st.error(f"Failed to download or load 'en_core_web_trf': {e_trf}")
            st.warning("Falling back to 'en_core_web_sm'. NER performance might be reduced.")
            # Fallback to small model
            try:
                nlp = spacy.load("en_core_web_sm")
                st.success("Loaded spaCy 'en_core_web_sm' model as fallback.")
                return nlp
            except OSError:
                st.warning("Could not load 'en_core_web_sm'. Downloading...")
                try:
                    spacy.cli.download("en_core_web_sm")
                    nlp = spacy.load("en_core_web_sm")
                    st.success("Successfully downloaded and loaded 'en_core_web_sm'.")
                    return nlp
                except Exception as e_sm:
                    st.error(f"Failed to download or load 'en_core_web_sm': {e_sm}")
                    st.error("Could not load any spaCy model. NER features will be disabled.")
                    return None # Indicate failure

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot proceed without model files.")
        return None, None
    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        st.success("DistilGPT2 model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        st.error("Please ensure all model files were downloaded correctly.")
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
    "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note space here, keep if intended
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
    # Replace dynamic first to avoid conflicts if dynamic values contain static keys
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    dynamic_placeholders = {}
    if nlp is None: # Handle case where spaCy model failed to load
        st.warning("spaCy model not available. Cannot extract dynamic entities like Event or City.", icon="⚠️")
    else:
        try:
            doc = nlp(user_question)
            for ent in doc.ents:
                # Simple check for common event/location labels
                if ent.label_ in ["EVENT", "WORK_OF_ART", "PRODUCT"]: # Added more potential event labels
                    event_text = ent.text.strip().title()
                    dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                elif ent.label_ in ["GPE", "LOC", "FAC"]: # Added more potential location labels
                    city_text = ent.text.strip().title()
                    dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
        except Exception as e:
            st.error(f"Error during NER processing: {e}")

    # Set defaults if not found
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "the event" # More generic default
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "your city" # More generic default
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"Instruction: {instruction} Response:"
    try:
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
        # Decode only the generated part, excluding the prompt
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        # Sometimes the model might still include "Response:" if not fully fine-tuned
        if response.startswith("Response:"):
            response = response[len("Response:"):].strip()
        elif response.startswith(" Response:"):
             response = response[len(" Response:"):].strip()

        return response.strip()
    except Exception as e:
        st.error(f"Error during model generation: {e}")
        return "Sorry, I encountered an error while generating the response."


# --- CSS Styling ---
st.markdown(
    """
<style>
/* General Button Styling */
.stButton>button {
    border: none;
    border-radius: 25px; /* Rounded corners */
    padding: 10px 20px; /* Padding */
    font-size: 1em; /* Adjusted font size slightly */
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
    color: white !important; /* Ensure text is white */
}
.stButton>button:hover {
    transform: scale(1.05); /* Slightly larger on hover */
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); /* Shadow on hover */
    color: white !important; /* Ensure text stays white on hover */
}
.stButton>button:active {
    transform: scale(0.98); /* Slightly smaller when clicked */
}

/* Specific Button Gradients */
/* Default/Continue/Reset Button */
.stButton:not(:has(button[kind="secondary"]))>button { /* Target default buttons */
    background: linear-gradient(90deg, #ff8a00, #e52e71); /* Orange to Pink gradient */
}
/* "Ask this question" Button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]:first-of-type > button { /* More specific selector */
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Blue gradient */
}

/* Reset Button Specific Style (if needed, otherwise it takes default) */
/* You can add specific styles for the reset button using its key if necessary */
/* Example: .stButton button[data-testid="stResetButton"] { background: ...; } */
/* Make sure to add data-testid="stResetButton" to the reset button element if you use this */

/* Apply Times New Roman to other elements */
body, .stApp, .stMarkdown, .stSelectbox, .stTextInput, .stTextArea, .stChatMessage, .stAlert, .stSpinner > div > div {
    font-family: 'Times New Roman', Times, serif !important;
}
/* Ensure chat message content also uses the font */
.stChatMessage [data-testid="stMarkdownContainer"] p,
.stChatMessage [data-testid="stMarkdownContainer"] ul,
.stChatMessage [data-testid="stMarkdownContainer"] li,
.stChatMessage [data-testid="stMarkdownContainer"] b {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Horizontal line separator */
.horizontal-line {
    border-top: 2px solid #e0e0e0; /* Adjust color and thickness */
    margin: 15px 0; /* Spacing */
}

/* Chat Input Shadow Effect */
div[data-testid="stChatInput"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Softer shadow */
    border: 1px solid #eee; /* Subtle border */
    border-radius: 8px; /* Slightly more rounded */
    padding: 8px 15px; /* Adjust padding */
    margin: 15px 0; /* Adjust margin */
    background-color: #ffffff; /* Ensure background is white */
}

/* Assistant message styling for response time */
.assistant-message-container {
    display: flex;
    flex-direction: column;
    align-items: flex-start; /* Align items to the start */
}
.response-time {
    font-size: 0.75em; /* Smaller font size for time */
    color: #888; /* Grey color for less emphasis */
    margin-top: -2px; /* Adjust spacing relative to the main message */
    margin-left: 2px; /* Indent slightly */
    font-style: italic;
}
.response-content {
    margin-top: 4px; /* Space between time and response */
}
</style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit App Logic ---

st.markdown("<h1 style='font-size: 43px;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

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

# Example queries
example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket for the upcoming music festival in London?", # Added example entities
    "How do I change my personal details on my ticket?",
    "How can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?",
    "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation status?",
    "How can I sell my ticket?",
    "Are there any concerts in Paris next month?" # Added example entities
]

# --- Model Loading Section ---
if not st.session_state.models_loaded:
    with st.spinner("Loading models and resources... This may take a moment..."):
        nlp_model = load_spacy_model()
        gpt_model, gpt_tokenizer = load_model_and_tokenizer()

        if gpt_model is not None and gpt_tokenizer is not None:
            st.session_state.models_loaded = True
            st.session_state.nlp = nlp_model # Can be None if spaCy failed
            st.session_state.model = gpt_model
            st.session_state.tokenizer = gpt_tokenizer
            # Don't rerun here, let the flow continue to disclaimer or chat
        else:
            st.error("Critical error: Failed to load the main language model. Chatbot cannot function.")
            # Keep models_loaded as False, the app won't proceed to chat

# --- Disclaimer Section ---
# Show disclaimer only if models ARE loaded and user hasn't clicked continue
if st.session_state.models_loaded and not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb;">
            <h2 style="font-size: 28px; color: #721c24; font-weight: bold; text-align: center; margin-bottom: 15px;">⚠️ Disclaimer</h2>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                This <b>Chatbot</b> assists with ticketing inquiries. Due to computational limits, it's fine-tuned for specific intents and might not handle all queries accurately.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                Optimized intents include:
            </p>
            <ul style="font-size: 16px; line-height: 1.6; color: #721c24; margin-left: 20px;">
                <li>Buy/Sell/Transfer/Upgrade/Find Ticket</li>
                <li>Cancel Ticket & Track Cancellation</li>
                <li>Change Personal Details</li>
                <li>Get Refund & Check Cancellation Fee</li>
                <li>Find Upcoming Events</li>
                <li>Customer Service & Ticket Information</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                Queries outside these areas may not receive accurate responses. We appreciate your understanding if the chatbot struggles sometimes.
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
            st.rerun() # Rerun to hide disclaimer and show chat

# --- Chat Interface Section ---
# Show chat only if models loaded AND user clicked continue
if st.session_state.models_loaded and st.session_state.show_chat:

    st.write("Ask me about ticket cancellations, refunds, event details, or other ticketing inquiries!")

    # Dropdown and Button section at the TOP
    col1, col2 = st.columns([4, 1]) # Adjust ratio if needed
    with col1:
        selected_query = st.selectbox(
            "Choose a query from examples:",
            ["Choose your question"] + example_queries,
            key="query_selectbox",
            label_visibility="collapsed"
        )
    with col2:
        # Use a more specific key for the button if needed for styling
        process_query_button = st.button("Ask this", key="query_button")

    # Access loaded models from session state (ensure they are loaded)
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    last_role = None # Track last message role for separator

    # Display chat messages from history
    for i, message in enumerate(st.session_state.chat_history):
        # Add separator line before a user message if the previous one was an assistant message
        if message["role"] == "user" and i > 0 and st.session_state.chat_history[i-1]["role"] == "assistant":
             st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

        with st.chat_message(message["role"], avatar=message["avatar"]):
            # Check if it's an assistant message to potentially include time
            if message["role"] == "assistant":
                 # Display the content which might already include the time string from generation
                 st.markdown(message["content"], unsafe_allow_html=True)
            else:
                 st.markdown(message["content"], unsafe_allow_html=True) # User message content

        last_role = message["role"]


    # --- Function to handle query processing (avoids code duplication) ---
    def handle_query(query):
        global last_role # Need to modify the global tracker
        # Capitalize first letter
        processed_query = query[0].upper() + query[1:] if query else query

        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": processed_query, "avatar": "👤"})
        if last_role == "assistant": # Check before displaying user message
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(processed_query, unsafe_allow_html=True)
        last_role = "user" # Update role *after* displaying user message

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            generating_response_text = "Generating response..."
            with st.spinner(generating_response_text):
                start_time = time.time() # Start timing
                dynamic_placeholders = extract_dynamic_placeholders(processed_query, nlp)
                response_gpt = generate_response(model, tokenizer, processed_query)
                end_time = time.time() # End timing
                elapsed_time = end_time - start_time
                time_str = f"({int(round(elapsed_time))}s)" # Format time

                full_response_text = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)

                # Combine time and response using the CSS classes
                styled_response = f"""
                <div class="assistant-message-container">
                    <span class="response-time">{time_str}</span>
                    <div class="response-content">{full_response_text}</div>
                </div>
                """

            message_placeholder.markdown(styled_response, unsafe_allow_html=True)

        # Add assistant message (including the styled HTML) to history
        st.session_state.chat_history.append({"role": "assistant", "content": styled_response, "avatar": "🤖"})
        last_role = "assistant" # Update role *after* displaying assistant message


    # --- Process selected query from dropdown ---
    if process_query_button:
        if selected_query == "Choose your question":
            st.toast("⚠️ Please select a question from the dropdown.", icon="💡")
        elif selected_query:
            handle_query(selected_query)
            # Clear the selectbox selection after processing
            st.session_state.query_selectbox = "Choose your question"
            st.rerun() # Rerun to update the selectbox and display the new messages properly


    # --- Process user input from chat box ---
    if prompt := st.chat_input("Enter your own question:"):
        if not prompt.strip():
            st.toast("⚠️ Please enter a question.", icon="✍️")
        else:
            handle_query(prompt)
            st.rerun() # Rerun to display the new messages immediately

    # --- Reset Button ---
    if st.session_state.chat_history:
        # Place reset button lower down or in a sidebar if preferred
        st.markdown("---") # Simple separator
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            last_role = None
            st.rerun()

# --- Fallback if models never loaded ---
elif not st.session_state.models_loaded and not st.session_state.show_chat:
    # This condition is met if loading failed in the first block
    st.error("Chatbot initialization failed due to model loading errors. Please check the logs above, ensure internet connectivity, and refresh the page.")
