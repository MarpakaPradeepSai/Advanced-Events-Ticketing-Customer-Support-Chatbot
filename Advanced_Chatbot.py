import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time
import threading # Needed for checking stop flag during generation (though true interruption is hard)

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
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(MODEL_FILES)

    for i, filename in enumerate(MODEL_FILES):
        url = f"{GITHUB_MODEL_URL}/{filename}"
        local_path = os.path.join(model_dir, filename)

        if not os.path.exists(local_path):
            status_text.text(f"Downloading {filename}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                file_size = int(response.headers.get('content-length', 0))
                chunk_size = 8192
                downloaded_size = 0

                with open(local_path, "wb") as f:
                     for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            # Optional: Add per-file progress if needed, but might slow down
                status_text.text(f"Downloaded {filename}.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from GitHub: {e}")
                # Clean up partially downloaded file if error occurs
                if os.path.exists(local_path):
                    os.remove(local_path)
                status_text.text("")
                progress_bar.empty()
                return False
        progress_bar.progress((i + 1) / total_files)

    status_text.text("Model files downloaded successfully.")
    time.sleep(1) # Keep success message visible briefly
    status_text.empty()
    progress_bar.empty()
    return True


# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        st.info("Downloading spaCy model (en_core_web_trf)...")
        spacy.cli.download("en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")
    return nlp

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading AI Model...")
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot proceed.")
        return None, None

    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {e}")
        st.error("Ensure all model files are present and not corrupted in /tmp/DistilGPT2_Model")
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
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE": # Geographical Political Entity (like cities, countries)
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
        # Add more entities if needed (DATE, ORG, etc.)
        # elif ent.label_ == "DATE":
        #     dynamic_placeholders['{{DATE}}'] = f"<b>{ent.text}</b>"

    # Provide default values if entities are not found
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "the event" # More generic default
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "your city" # More generic default
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
# NOTE: True interruption of model.generate is complex.
# This function now accepts a threading.Event to signal stopping.
# However, model.generate itself doesn't check this event mid-generation.
# The check happens *after* generation completes or *before* it starts in the Streamlit flow.
def generate_response(model, tokenizer, instruction, stop_event, max_length=256):
    # Check if stop was requested *before* even starting generation in this call
    if stop_event.is_set():
        return "Generation stopped."

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length // 2).to(device) # Truncate input to leave space for response

    response = ""
    try:
        with torch.no_grad():
            # The generate call itself is blocking and usually cannot be interrupted easily from another thread without OS-level signals or complex async setups.
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

        # Check stop_event *after* generation finishes
        if stop_event.is_set():
            return "Generation stopped."

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = response.find("Response:") + len("Response:")
        response = response[response_start:].strip()

    except Exception as e:
        st.error(f"Error during model generation: {e}")
        response = "Sorry, I encountered an error while generating the response."
    finally:
        # Move model back to CPU if necessary to free up GPU memory, depends on usage pattern
        # model.to("cpu")
        pass

    # Final check before returning
    if stop_event.is_set():
        return "Generation stopped."
    return response

# --- Initialize Session State ---
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "stop_event" not in st.session_state:
    # Use threading.Event for stop signal
    st.session_state.stop_event = threading.Event()
if 'current_input' not in st.session_state:
    st.session_state.current_input = ""


# --- CSS Styling ---
st.markdown(
    """
<style>
/* General Button Style */
.stButton>button {
    background: linear-gradient(90deg, #ff8a00, #e52e71); /* Stylish gradient */
    color: white !important; /* Ensure text is white */
    border: none;
    border-radius: 25px; /* Rounded corners */
    padding: 10px 20px; /* Padding */
    font-size: 1.1em; /* Adjusted Font size */
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

/* Times New Roman Font */
* {
    font-family: 'Times New Roman', Times, serif !important;
}
/* Adjust specific elements if needed */
.stSelectbox, .stTextInput, .stTextArea, .stChatMessage, .stAlert, .streamlit-expanderContent {
    font-family: 'Times New Roman', Times, serif !important;
}
/* Specific 'Ask this question' Button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */
}

/* Stop Generating Button Style */
div[data-testid="stButton"] button[kind="secondary"] {
    background-color: #f44336 !important; /* Red background */
    color: white !important;
    border: 1px solid #d32f2f !important;
    border-radius: 5px !important; /* Less rounded */
    padding: 3px 8px !important; /* Smaller padding */
    font-size: 0.8em !important; /* Smaller font size */
    font-weight: bold;
    margin-left: 10px; /* Space from the generating text */
    height: 28px; /* Fixed height */
    min-width: 60px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    line-height: 1; /* Adjust line height */
    font-family: 'Times New Roman', Times, serif !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background-color: #d32f2f !important; /* Darker red on hover */
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2);
    transform: scale(1.03);
}
div[data-testid="stButton"] button[kind="secondary"]:active {
    transform: scale(0.98);
}


/* Horizontal Line Separator */
.horizontal-line {
    border-top: 2px solid #e0e0e0;
    margin: 15px 0;
}

/* Chat Input Shadow Effect */
div[data-testid="stChatInput"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); /* Softer shadow */
    border-radius: 8px; /* Slightly more rounded */
    padding: 10px 15px; /* Adjust padding */
    margin: 10px 0;
    border: 1px solid #eee; /* Subtle border */
}

/* Style for Generating text + Stop button container */
.generating-container {
    display: flex;
    align-items: center;
    justify-content: space-between; /* Pushes button to the right */
    width: 100%;
}
.generating-text {
    font-style: italic;
    color: #555;
    margin-right: 10px; /* Space between text and button if not using space-between */
}
</style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.markdown("<h1 style='font-size: 43px;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize models only once
nlp = load_spacy_model()
model, tokenizer = load_model_and_tokenizer()

# Display Disclaimer and Continue button if chat hasn't started
if not st.session_state.show_chat:
    st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb;">
            <h1 style="font-size: 36px; color: #721c24; font-weight: bold; text-align: center;">⚠️Disclaimer</h1>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents, and may not be able to respond accurately to all types of queries.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                The chatbot is optimized to handle the following intents:
            </p>
            <ul style="font-size: 16px; line-height: 1.6; color: #721c24;">
                <li>Cancel Ticket</li> <li>Buy Ticket</li> <li>Sell Ticket</li> <li>Transfer Ticket</li> <li>Upgrade Ticket</li> <li>Find Ticket</li> <li>Change Personal Details on Ticket</li> <li>Get Refund</li> <li>Find Upcoming Events</li> <li>Customer Service</li> <li>Check Cancellation Fee</li> <li>Track Cancellation</li> <li>Ticket Information</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24;">
                Please note that this chatbot may not be able to assist with queries outside of these predefined intents.
                Even if the model fails to provide accurate responses from the predefined intents, we kindly ask for your patience and encourage you to try again.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Continue button aligned to the right using columns
    _, col2 = st.columns([4, 1])  # Adjust ratios as needed
    with col2:
        if st.button("Continue", key="continue_button"):
            st.session_state.show_chat = True
            st.rerun() # Rerun to show the chat interface

# Show chat interface only after clicking Continue and models are loaded
if st.session_state.show_chat:
    if model is None or tokenizer is None or nlp is None:
        st.error("Models could not be loaded. Please check the logs and ensure files are downloaded correctly. Refresh the page to try again.")
        st.stop() # Stop execution if models aren't ready

    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # Dropdown and Button section at the TOP
    col1, col2 = st.columns([3, 1]) # Columns for dropdown and button
    with col1:
        example_queries = [
            "Choose your question", # Placeholder
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
        selected_query = st.selectbox(
            "Choose a query from examples:",
            example_queries,
            key="query_selectbox",
            index=0, # Default to placeholder
            label_visibility="collapsed" # Hide label for cleaner look
        )
    with col2:
        # Disable button if generating or if placeholder selected
        process_query_button = st.button(
            "Ask this question",
            key="query_button",
            disabled=st.session_state.is_generating or selected_query == "Choose your question"
            )

    # --- Chat History Display ---
    last_role = None
    for i, message in enumerate(st.session_state.chat_history):
        is_last_message = i == len(st.session_state.chat_history) - 1
        # Add separator line logic if needed (currently handled differently)
        # if message["role"] == "user" and last_role == "assistant":
        #     st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

        with st.chat_message(message["role"], avatar=message["avatar"]):
            # If it's the last message and it's an assistant currently generating
            if is_last_message and message["role"] == "assistant" and st.session_state.is_generating:
                # Container for generating text and stop button
                gen_container = st.container()
                with gen_container:
                    col1_gen, col2_gen = st.columns([0.85, 0.15]) # Adjust ratio for text and button
                    with col1_gen:
                        st.markdown('<span class="generating-text">Generating response...</span>', unsafe_allow_html=True)
                    with col2_gen:
                        # Use a secondary button style for Stop
                        if st.button("⏹️ Stop", key=f"stop_gen_{i}", type="secondary"):
                            st.session_state.stop_event.set() # Signal the generation to stop
                            st.session_state.is_generating = False # Update state immediately
                            # Update the message content directly
                            st.session_state.chat_history[-1]['content'] = "🛑 Generation stopped by user."
                            st.rerun() # Rerun to redraw the message without the button

            # Otherwise, display the content normally
            else:
                st.markdown(message["content"], unsafe_allow_html=True)
        last_role = message["role"]


    # --- Function to handle query processing ---
    def handle_query(prompt):
        if not prompt or prompt == "Choose your question":
            st.toast("⚠️ Please enter or select a question.")
            return

        prompt_formatted = prompt[0].upper() + prompt[1:] if prompt else prompt
        st.session_state.current_input = prompt # Store the input that triggered generation

        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": prompt_formatted, "avatar": "👤"})

        # Append a placeholder for the assistant's response and set generating flag
        st.session_state.chat_history.append({"role": "assistant", "content": "...", "avatar": "🤖"})
        st.session_state.is_generating = True
        st.session_state.stop_event.clear() # Reset stop event for the new generation

        # Rerun to display the user message and the "Generating..." placeholder with Stop button
        st.rerun()


    # --- Process selected query from dropdown ---
    if process_query_button and selected_query != "Choose your question":
        handle_query(selected_query)


    # --- Process text input ---
    if prompt := st.chat_input("Enter your own question:", key="chat_input_box", disabled=st.session_state.is_generating):
         handle_query(prompt)


    # --- Generation Logic (runs after rerun if is_generating is true) ---
    if st.session_state.is_generating:
        # Get the prompt that triggered this generation
        prompt_to_process = st.session_state.current_input

        # Ensure we have a valid prompt before proceeding
        if prompt_to_process:
            try:
                # Extract dynamic placeholders (do this before generation)
                dynamic_placeholders = extract_dynamic_placeholders(prompt_to_process, nlp)

                # Generate response
                response_gpt = generate_response(model, tokenizer, prompt_to_process, st.session_state.stop_event)

                # Check if stopped *during* or *after* generation
                if st.session_state.stop_event.is_set():
                    full_response = "🛑 Generation stopped by user."
                elif response_gpt == "Generation stopped.": # Handle case where generate_response returned stop message
                     full_response = "🛑 Generation stopped by user."
                elif response_gpt:
                    full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                else:
                    full_response = "Sorry, I couldn't generate a response." # Fallback

            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response = "An error occurred while processing your request."
            finally:
                # Update the last message in history (which was the placeholder)
                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                    st.session_state.chat_history[-1]["content"] = full_response

                # Reset generation state
                st.session_state.is_generating = False
                st.session_state.stop_event.clear() # Clear the event for the next run
                st.session_state.current_input = "" # Clear the stored input
                st.rerun() # Rerun to display the final response and remove the stop button


    # Conditionally display reset button
    if st.session_state.chat_history:
        if st.button("Reset Chat", key="reset_button", disabled=st.session_state.is_generating):
            st.session_state.chat_history = []
            st.session_state.is_generating = False
            st.session_state.stop_event.clear()
            st.session_state.current_input = ""
            st.rerun()
