import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time

# --- Configuration ---
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
MODEL_DIR = "/tmp/DistilGPT2_Model" # Define model directory path

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
    "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note: Extra space in original key
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

# --- Functions ---
# Function to download model files from GitHub
def download_model_files(model_dir=MODEL_DIR):
    if not os.path.exists(model_dir):
        st.info(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
    else:
        st.info(f"Model directory already exists: {model_dir}")

    all_files_exist = True
    for filename in MODEL_FILES:
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(local_path):
            all_files_exist = False
            st.warning(f"File not found: {local_path}. Downloading...")
            url = f"{GITHUB_MODEL_URL}/{filename}"
            try:
                response = requests.get(url, stream=True, timeout=30) # Use stream and timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success(f"Downloaded {filename}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from {url}. Error: {e}")
                # Optionally remove partially downloaded file
                if os.path.exists(local_path):
                    os.remove(local_path)
                return False # Stop download process if one file fails
        # else:
            # st.info(f"File already exists: {filename}") # Can be verbose, uncomment if needed

    if all_files_exist and len(os.listdir(model_dir)) >= len(MODEL_FILES):
         st.success("All model files are present.")
         return True
    elif not all_files_exist:
         st.error("Model download failed for one or more files.")
         return False
    else:
         st.warning("Some files were present, others downloaded. Check logs.")
         return True # Allow proceeding if downloads seemed successful


# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_trf"
    try:
        # Try loading the model directly
        nlp = spacy.load(model_name)
        st.success(f"Successfully loaded spaCy model '{model_name}'.")
        return nlp
    except OSError:
        # If not found, download it
        st.warning(f"spaCy model '{model_name}' not found. Downloading...")
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
            st.success(f"Successfully downloaded and loaded spaCy model '{model_name}'.")
            return nlp
        except Exception as e:
            st.error(f"Failed to download or load spaCy model '{model_name}'. Error: {e}")
            st.error("NER features for dynamic placeholders like EVENT and CITY will not work.")
            return None # Return None if loading fails


# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading AI Model...")
def load_model_and_tokenizer():
    model_dir = MODEL_DIR
    # Ensure files are downloaded before trying to load
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot load the model.")
        return None, None

    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        st.success("AI Model and Tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_dir}: {e}")
        st.error("Please ensure all model files were downloaded correctly and are not corrupted.")
        return None, None

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    # Apply dynamic placeholders last, in case static ones contained dynamic patterns
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    dynamic_placeholders = {}
    # Set defaults first
    dynamic_placeholders['{{EVENT}}'] = "event"
    dynamic_placeholders['{{CITY}}'] = "city"

    if nlp and user_question: # Check if nlp model loaded and question is not empty
        try:
            doc = nlp(user_question)
            for ent in doc.ents:
                if ent.label_ == "EVENT":
                    event_text = ent.text.title()
                    dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                elif ent.label_ == "GPE": # GPE typically refers to geopolitical entities (cities, countries)
                    city_text = ent.text.title()
                    dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
        except Exception as e:
            st.warning(f"NER processing failed: {e}. Using default placeholders.")
    elif not nlp:
         st.warning("SpaCy model not loaded. Using default placeholders for EVENT and CITY.")

    return dynamic_placeholders


# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    if not model or not tokenizer:
        return "Error: Model or Tokenizer not loaded."

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Format the prompt correctly, expecting the model to continue after "Response:"
    input_text = f"Instruction: {instruction}\nResponse:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device) # Add truncation

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
        # Decode only the newly generated tokens
        response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Clean up potential artifacts (sometimes models repeat the prompt start)
        # if response.startswith("Response:"):
        #     response = response[len("Response:"):].strip()
        # If the model didn't generate anything meaningful, return a default
        if not response.strip():
             return "I am sorry, I couldn't generate a specific response for that. Can you please rephrase or ask something else?"

        return response.strip()

    except Exception as e:
        st.error(f"Error during response generation: {e}")
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
    background: linear-gradient(90deg, #ff8a00, #e52e71); /* Default gradient for Reset */
    color: white !important;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 1.1em; /* Slightly smaller */
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: 5px;
    width: auto;
    min-width: 100px;
    font-family: 'Times New Roman', Times, serif !important; /* Ensure font */
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
    color: white !important;
}
.stButton>button:active {
    transform: scale(0.98);
}

/* Specific Styling for 'Ask this question' button */
/* Targets the button within the horizontal block likely containing the selectbox and button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #29ABE2, #0077B6) !important; /* Blue gradient */
    color: white !important;
    font-size: 1.1em !important; /* Match size */
    /* Inherits other styles like padding, border-radius etc. from the general .stButton>button */
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
    transform: scale(1.05); /* Keep hover effect */
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
    color: white !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:active {
    transform: scale(0.98); /* Keep active effect */
}


/* Horizontal Line Separator */
.horizontal-line {
    border-top: 1px solid #e0e0e0; /* Thinner line */
    margin: 10px 0; /* Adjust spacing */
}

/* Chat Input Shadow Effect */
div[data-testid="stChatInput"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Softer shadow */
    border-radius: 8px; /* Slightly more rounded */
    padding: 5px 10px; /* Adjust padding */
    margin: 15px 0 10px 0; /* Adjust margins */
    border: 1px solid #eee; /* Subtle border */
}

/* Disclaimer Box */
.disclaimer-box {
    background-color: #fff3cd; /* Lighter yellow */
    padding: 20px;
    border-radius: 10px;
    color: #664d03; /* Darker text for contrast */
    border: 1px solid #ffe69c;
    font-family: 'Times New Roman', Times, serif !important;
    margin-bottom: 20px; /* Space below disclaimer */
}
.disclaimer-box h1 {
    font-size: 28px; /* Smaller heading */
    color: #664d03;
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
}
.disclaimer-box p, .disclaimer-box ul {
    font-size: 15px; /* Slightly smaller text */
    line-height: 1.6;
    color: #664d03;
}
.disclaimer-box ul {
    list-style-position: inside;
    padding-left: 10px;
}
.disclaimer-box b {
    font-weight: bold;
    color: #5c4001; /* Slightly darker bold */
}

/* Adjust Continue button alignment */
div[data-testid="stHorizontalBlock"].st-emotion-cache-180ud1x { /* Adjust class if needed */
    justify-content: flex-end; /* Push content (button) to the right */
}

/* Ensure chat messages use the font */
.stChatMessage {
    font-family: 'Times New Roman', Times, serif !important;
}

</style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit UI ---
st.markdown("<h1 style='font-size: 38px; text-align: center;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# --- Session State Initialization ---
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_to_process" not in st.session_state:
    st.session_state.query_to_process = None # To store query from dropdown button click
if "last_role" not in st.session_state:
    st.session_state.last_role = None # Track last message role for separators

# --- Disclaimer Section ---
if not st.session_state.show_chat:
    st.markdown(
        """
        <div class="disclaimer-box">
            <h1>⚠️ Disclaimer</h1>
            <p>
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents and may not be able to respond accurately to all types of queries.
            </p>
            <p>The chatbot is optimized to handle the following intents:</p>
            <ul>
                <li>Cancel Ticket</li><li>Buy Ticket</li><li>Sell Ticket</li><li>Transfer Ticket</li>
                <li>Upgrade Ticket</li><li>Find Ticket</li><li>Change Personal Details on Ticket</li>
                <li>Get Refund</li><li>Find Upcoming Events</li><li>Customer Service</li>
                <li>Check Cancellation Fee</li><li>Track Cancellation</li><li>Ticket Information</li>
            </ul>
            <p>
                Please note that this chatbot may not be able to assist with queries outside of these predefined intents.
                Even if the model fails to provide accurate responses for the predefined intents, we kindly ask for your patience and encourage you to try again or rephrase your question.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Continue button aligned to the right
    # Create columns for alignment: one large empty column, one small for the button
    _, col2 = st.columns([4, 1]) # Adjust ratio if needed
    with col2:
        if st.button("Continue", key="continue_button", help="Start interacting with the chatbot"):
            st.session_state.show_chat = True
            st.rerun() # Rerun to hide disclaimer and show chat

# --- Chat Interface Section ---
if st.session_state.show_chat:

    # --- Load Models (only once after clicking Continue) ---
    # Initialize spaCy model for NER
    nlp = load_spacy_model()

    # Load DistilGPT2 model and tokenizer
    # Show spinner here during the potentially long loading process
    model, tokenizer = load_model_and_tokenizer()

    if model is None or tokenizer is None:
        st.error("Critical error: Failed to load the AI model. The chatbot cannot function.")
        st.stop() # Stop execution if model loading failed

    st.markdown("---") # Separator line
    st.write("Ask me about ticket cancellations, refunds, upcoming events, or other related inquiries!")

    # --- Example Queries Section ---
    example_queries = [
        "How do I buy a ticket?",
        "How can I upgrade my ticket for the upcoming concert in London?", # Added example with entity
        "How do I change my personal details on my ticket?",
        "How can I find details about upcoming events?",
        "How do I contact customer service?",
        "How do I get a refund?",
        "What is the ticket cancellation fee?",
        "How can I track my ticket cancellation?",
        "How can I sell my ticket?"
    ]

    # Use columns for better layout of selectbox and button
    col1, col2 = st.columns([3, 1]) # Adjust ratio: 3 parts for selectbox, 1 for button

    with col1:
        selected_query = st.selectbox(
            "Or choose an example question:",
            [""] + example_queries, # Add empty option
            index=0, # Default to empty
            key="query_selectbox",
            label_visibility="collapsed"
        )

    with col2:
        # Button to process the selected query
        process_query_button = st.button(
            "Ask this", # Shorter text
            key="query_button",
            help="Submit the selected example question"
            )

    # --- Logic to handle the "Ask this" button click ---
    if process_query_button:
        if selected_query: # Check if a valid query is selected
            # Store the selected query to be processed in the next rerun
            st.session_state.query_to_process = selected_query
            # Clear the selectbox selection after storing
            st.session_state.query_selectbox = ""
            # Trigger immediate rerun to process the stored query
            st.rerun()
        else:
            st.toast("⚠️ Please select an example question first.", icon="💡")

    st.markdown("---") # Separator before chat history

    # --- Process Stored Query (from dropdown button) ---
    # This block runs *after* the button click caused a rerun
    if st.session_state.query_to_process:
        query = st.session_state.query_to_process
        query = query[0].upper() + query[1:] if query else query # Capitalize

        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query, "avatar": "👤"})
        # Add assistant response placeholder (will be filled below)

        # Display the user message immediately
        if st.session_state.last_role == "assistant":
             st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
             st.markdown(query, unsafe_allow_html=True)
        st.session_state.last_role = "user"

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            generating_response_text = "Generating response..."
            full_response = "Thinking..." # Initial placeholder text
            message_placeholder.markdown(f"{full_response} ⏳", unsafe_allow_html=True)

            try:
                # Use spinner context for visual feedback during generation
                with st.spinner(generating_response_text):
                    dynamic_placeholders = extract_dynamic_placeholders(query, nlp)
                    raw_response = generate_response(model, tokenizer, query)
                    full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)
                    # time.sleep(1) # Simulate delay if needed for testing

                # Update the placeholder with the final response
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                # Add the *final* assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
                st.session_state.last_role = "assistant"

            except Exception as e:
                 st.error(f"An error occurred: {e}")
                 error_message = "Sorry, I encountered an error. Please try again."
                 message_placeholder.markdown(error_message, unsafe_allow_html=True)
                 # Add error message to history
                 st.session_state.chat_history.append({"role": "assistant", "content": error_message, "avatar": "🤖"})
                 st.session_state.last_role = "assistant"


        # IMPORTANT: Clear the query_to_process state variable
        st.session_state.query_to_process = None
        # Rerun *again* to ensure the chat history display updates correctly *without* reprocessing
        st.rerun()

    # --- Display Chat History ---
    # This needs to run *after* potential processing of query_to_process
    st.markdown("#### Chat History")
    chat_container = st.container() # Use a container for chat messages
    with chat_container:
        current_last_role = None # Use a local var for this loop run
        for i, message in enumerate(st.session_state.chat_history):
            # Add separator only between user and assistant messages
            if i > 0 and message["role"] == "user" and st.session_state.chat_history[i-1]["role"] == "assistant":
                 st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)

            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"], unsafe_allow_html=True)
            current_last_role = message["role"] # Keep track of the role within the loop

        # Update the session state last_role based on the actual last message displayed
        st.session_state.last_role = current_last_role


    # --- Chat Input Section (at the bottom) ---
    if prompt := st.chat_input("Enter your own question:"):
        prompt = prompt.strip()
        if not prompt:
            st.toast("⚠️ Please enter a question.", icon="⌨️")
        else:
            prompt_capitalized = prompt[0].upper() + prompt[1:] if prompt else prompt

            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt_capitalized, "avatar": "👤"})

            # Display user message immediately (causes a mini-rerun)
            # No need for manual display here, chat_input rerun handles it.

            # Add placeholder for assistant message
            st.session_state.chat_history.append({"role": "assistant", "content": "Thinking... ⏳", "avatar": "🤖"})

            # Trigger rerun to display user message and start generation
            st.rerun() # Rerun immediately

    # --- Process Chat Input (if last message is from user or placeholder) ---
    # Check if the last message needs a response generated
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant" and st.session_state.chat_history[-1]["content"] == "Thinking... ⏳":
        # Get the corresponding user prompt
        user_prompt_index = -2 # The user prompt is before the placeholder
        if len(st.session_state.chat_history) >= 2 and st.session_state.chat_history[user_prompt_index]["role"] == "user":
            user_prompt = st.session_state.chat_history[user_prompt_index]["content"]

            # Generate response
            try:
                # Display the placeholder while generating
                # The placeholder is already in history, the rerun will show it
                dynamic_placeholders = extract_dynamic_placeholders(user_prompt, nlp)
                raw_response = generate_response(model, tokenizer, user_prompt)
                full_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)
                # time.sleep(1) # Optional delay

                # Update the last message (the placeholder) with the actual response
                st.session_state.chat_history[-1]["content"] = full_response
                st.session_state.last_role = "assistant"

            except Exception as e:
                 st.error(f"An error occurred: {e}")
                 error_message = "Sorry, I encountered an error. Please try again."
                 # Update the last message with the error
                 st.session_state.chat_history[-1]["content"] = error_message
                 st.session_state.last_role = "assistant"

            # Rerun to display the generated response
            st.rerun()


    # --- Reset Button ---
    if st.session_state.chat_history:
        if st.button("Reset Chat", key="reset_button", help="Clear the chat history"):
            st.session_state.chat_history = []
            st.session_state.last_role = None
            st.session_state.query_to_process = None # Also clear any pending query
            st.rerun()
