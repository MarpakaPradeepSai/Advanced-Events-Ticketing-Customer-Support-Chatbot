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
            # Add progress indication for download
            with st.spinner(f"Downloading {filename}..."):
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    with open(local_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to download {filename} from GitHub: {e}")
                    # Attempt to remove potentially incomplete file
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    return False
    return True

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    # Add progress indication
    with st.spinner("Loading language model (spaCy)..."):
        try:
            # Check if model is already downloaded, if not, download it
            spacy.cli.download("en_core_web_trf")
            nlp = spacy.load("en_core_web_trf")
        except Exception as e:
            st.error(f"Failed to load spaCy model 'en_core_web_trf'. Error: {e}")
            st.info("Attempting to download the model. This might take a moment.")
            try:
                spacy.cli.download("en_core_web_trf")
                nlp = spacy.load("en_core_web_trf")
            except Exception as download_e:
                st.error(f"Failed to download and load spaCy model after retry. Error: {download_e}")
                return None
    return nlp


# Load the DistilGPT2 model and tokenizer
# Added show_spinner=True for better user feedback
@st.cache_resource(show_spinner=True, hash_funcs={GPT2LMHeadModel: id, GPT2Tokenizer: id})
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    st.write("Checking for model files...") # Feedback before potential download
    if not download_model_files(model_dir):
        st.error("Model download failed. Cannot proceed.")
        return None, None

    st.write("Model files downloaded/verified. Loading model and tokenizer...") # Feedback after download
    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        st.write("Model and tokenizer loaded successfully.") # Success feedback
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model/tokenizer from {model_dir}. Error: {e}")
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
    "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note: Extra space here in original, kept it
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
    # Ensure nlp model is loaded
    if nlp is None:
        st.warning("Language model (spaCy) not loaded. Cannot extract dynamic entities.")
        return {'{{EVENT}}': "event", '{{CITY}}': "city"}

    doc = nlp(user_question)
    dynamic_placeholders = {}
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE": # GPE usually means Geopolitical Entity (cities, states, countries)
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
        # Add more entity types if needed, e.g., DATE, TIME, ORG
        # elif ent.label_ == "DATE":
        #     date_text = ent.text
        #     dynamic_placeholders['{{DATE}}'] = f"<b>{date_text}</b>"

    # Provide default values if entities are not found
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event" # Use a generic default
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city" # Use a generic default
    return dynamic_placeholders


# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    # Ensure model and tokenizer are loaded
    if model is None or tokenizer is None:
        return "Error: Model or Tokenizer not available."

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"Instruction: {instruction} Response:"
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length).to(device) # Added truncation
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length, # Use max_new_tokens instead of max_length for clearer control over output length
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        # Decode only the generated part, excluding the prompt
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, I encountered an error while generating the response."


# --- CSS Styling ---
st.markdown(
    """
<style>
/* General Button Style (from original code) */
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
* {
    font-family: 'Times New Roman', Times, serif !important;
}

/* Specific adjustments for Streamlit elements (examples) */
.stSelectbox > div > div > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stChatMessage,
.streamlit-expanderHeader, /* Expander headers */
.streamlit-expanderContent, /* Expander content */
.stAlert /* Alerts like st.error, st.warning */
{
    font-family: 'Times New Roman', Times, serif !important;
}

/* Style for the 'Ask this question' button (overrides general button style) */
/* Targets the button specifically within the horizontal block often used for selectbox+button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) {
    background: linear-gradient(90deg, #29ABE2, #0077B6); /* Different gradient */
    color: white !important;
    /* Inherits other properties like padding, border-radius, font-size etc. from .stButton>button */
}

/* Style for the horizontal line separator */
.horizontal-line {
    border-top: 2px solid #e0e0e0; /* Adjust color and thickness */
    margin: 15px 0; /* Adjust spacing */
}

/* --- NEW CSS RULE TO ALIGN CONTINUE BUTTON --- */
/* This targets the div wrapping the 'Continue' button */
.continue-button-container {
    text-align: right; /* Aligns inline or inline-block children (like the button) to the right */
    margin-top: 15px; /* Add some space above the button */
}
/* Apply general button styling to the continue button specifically if needed, */
/* but it should inherit from the global .stButton>button rule */
/* .continue-button-container .stButton>button { ... specific styles if needed ... } */

</style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit UI ---
st.markdown("<h1 style='font-size: 43px; font-family: 'Times New Roman', Times, serif;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# Initialize session state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_role" not in st.session_state:
    st.session_state.last_role = None
if "nlp_model" not in st.session_state:
    st.session_state.nlp_model = None
if "gpt_model" not in st.session_state:
    st.session_state.gpt_model = None
if "gpt_tokenizer" not in st.session_state:
    st.session_state.gpt_tokenizer = None


# --- Disclaimer and Initial Load ---
if not st.session_state.show_chat:
    st.markdown( # Disclaimer content
        """
        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; color: #721c24; border: 1px solid #f5c6cb; font-family: 'Times New Roman', Times, serif;">
            <h1 style="font-size: 36px; color: #721c24; font-weight: bold; text-align: center; font-family: 'Times New Roman', Times, serif;">⚠️Disclaimer</h1>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24; font-family: 'Times New Roman', Times, serif;">
                This <b>Chatbot</b> has been designed to assist users with a variety of ticketing-related inquiries. However, due to computational limitations, this model has been fine-tuned on a select set of intents, and may not be able to respond accurately to all types of queries.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #721c24; font-family: 'Times New Roman', Times, serif;">
                The chatbot is optimized to handle the following intents:
            </p>
            <ul style="font-size: 16px; line-height: 1.6; color: #721c24; margin-left: 20px; font-family: 'Times New Roman', Times, serif;">
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
                Please note that this chatbot may not be able to assist with queries outside of these predefined intents.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Continue Button Section (Aligned Right) ---
    # Wrap the button in the div with the new class for alignment
    st.markdown('<div class="continue-button-container">', unsafe_allow_html=True)
    if st.button("Continue", key="continue_button"):
        # Load models only when Continue is clicked
        with st.spinner("Initializing chatbot components... Please wait."):
            st.session_state.nlp_model = load_spacy_model()
            st.session_state.gpt_model, st.session_state.gpt_tokenizer = load_model_and_tokenizer()

        # Check if models loaded successfully before showing chat
        if st.session_state.nlp_model and st.session_state.gpt_model and st.session_state.gpt_tokenizer:
            st.session_state.show_chat = True
            st.rerun() # Rerun to hide disclaimer and show chat interface
        else:
            st.error("Failed to initialize chatbot components. Please check the logs or try refreshing.")
            # Keep show_chat as False
    st.markdown('</div>', unsafe_allow_html=True)


# --- Chat Interface (Shown after clicking Continue and successful load) ---
if st.session_state.show_chat:
    # Ensure models are loaded before proceeding
    if not st.session_state.nlp_model or not st.session_state.gpt_model or not st.session_state.gpt_tokenizer:
        st.error("Chatbot components are not loaded correctly. Please try refreshing the page.")
        st.stop() # Stop execution if models aren't ready

    st.markdown("<p style='font-family: Times New Roman, Times, serif;'>Ask me about ticket cancellations, refunds, or any event-related inquiries!</p>", unsafe_allow_html=True)

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

    # --- Dropdown and Button for Example Queries ---
    col_select, col_button = st.columns([4, 1]) # Adjust ratio as needed
    with col_select:
        selected_query = st.selectbox(
            "Choose a query from examples:",
            ["Choose your question"] + example_queries,
            key="query_selectbox",
            label_visibility="collapsed" # Hides the label "Choose a query..."
        )
    with col_button:
        # Use a unique key and apply specific styling via CSS rule defined earlier
        process_query_button = st.button("Ask selected", key="query_button")


    # --- Display Chat History ---
    for message in st.session_state.chat_history:
        # Add separator line if the last message was from the assistant
        if message["role"] == "user" and st.session_state.last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        st.session_state.last_role = message["role"] # Update last role after displaying


    # --- Process Selected Query ---
    if process_query_button:
        if selected_query == "Choose your question":
            st.toast("⚠️ Please select a question from the dropdown first.", icon="⚠️")
        elif selected_query:
            prompt_from_dropdown = selected_query
            # Capitalize first letter
            prompt_from_dropdown = prompt_from_dropdown[0].upper() + prompt_from_dropdown[1:] if prompt_from_dropdown else prompt_from_dropdown

            # Add user message to history and display it
            st.session_state.chat_history.append({"role": "user", "content": prompt_from_dropdown, "avatar": "👤"})
            if st.session_state.last_role == "assistant":
                 st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt_from_dropdown, unsafe_allow_html=True)
            st.session_state.last_role = "user"

            # Generate and display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                with st.spinner("Generating response..."):
                    dynamic_placeholders = extract_dynamic_placeholders(prompt_from_dropdown, st.session_state.nlp_model)
                    response_gpt = generate_response(st.session_state.gpt_model, st.session_state.gpt_tokenizer, prompt_from_dropdown)
                    full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                    # time.sleep(1) # Optional artificial delay

                message_placeholder.markdown(full_response, unsafe_allow_html=True)
            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            st.session_state.last_role = "assistant"
            # Rerun to clear the selection box implicitly if desired, or just update state
            st.rerun()


    # --- Process User Input from Chat Input ---
    if prompt := st.chat_input("Enter your own question:"):
        # Capitalize first letter
        prompt = prompt[0].upper() + prompt[1:] if prompt else prompt

        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        if st.session_state.last_role == "assistant":
             st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
        st.session_state.last_role = "user"

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Generating response..."):
                dynamic_placeholders = extract_dynamic_placeholders(prompt, st.session_state.nlp_model)
                response_gpt = generate_response(st.session_state.gpt_model, st.session_state.gpt_tokenizer, prompt)
                full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                # time.sleep(1) # Optional artificial delay

            message_placeholder.markdown(full_response, unsafe_allow_html=True)
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        st.session_state.last_role = "assistant"
        # Rerun to ensure the input box clears and the chat updates smoothly
        st.rerun()


    # --- Reset Chat Button ---
    if st.session_state.chat_history:
        # Place reset button maybe less prominently, e.g., in sidebar or at bottom
        st.markdown("---") # Separator
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
            st.session_state.last_role = None
            st.rerun() # Rerun to clear the chat display
