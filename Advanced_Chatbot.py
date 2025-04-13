import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time

# --- Constants and Setup ---
GITHUB_MODEL_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Tick.content)
                st.info(f"Downloaded {filename} successfully.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename} from GitHub: {e}")
                # Attempt to remove partially downloaded file if it exists
                ifeting-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"
MODEL_FILES = [
    "config.json", "generation_config. os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass #json", "merges.txt",
    "model.safetensors", "special_tokens_map.json",
    "tokenizer_config.json", "vocab.json"
]
MODEL_DIR = "/ Ignore errors during cleanup
                return False # Indicate download failure
    if all_files_present:tmp/DistilGPT2_Model" # Using /tmp for compatibility with Streamlit Cloud

# --- Functions ---

# Function to download model files from GitHub
def download_model_files(
         st.info("Model files already present.")
    return True

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model, attemptsmodel_dir=MODEL_DIR):
    """Downloads model files if they don't exist locally."""
    os.makedirs(model_dir, download if necessary, includes fallback."""
    model_name_trf = "en_core_ exist_ok=True)
    all_files_present = True
    for filename in MODELweb_trf"
    model_name_sm = "en__FILES:
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(localcore_web_sm"
    nlp = None
    try:
        _path):
            all_files_present = False
            url = f"{GITHUB_MODEL_URL}/{filename}"
            st.info(f"Downloading {filenamenlp = spacy.load(model_name_trf)
        st.success}...")
            try:
                response = requests.get(url, timeout=60) # Increased timeout
                response.raise_for_status() # Raise HTTPError for(f"Loaded spaCy model '{model_name_trf}'.")
        return nlp
     bad responses (4xx or 5xx)
                with open(local_path, "wb") as f:
                    f.write(responseexcept OSError:
        st.warning(f"SpaCy model '{model_name_trf}' not found. Attempting to download...")
        try:
            .content)
                st.info(f"Downloaded {filename} successfully.")
            except requests.exceptionsspacy.cli.download(model_name_trf)
            nlp = spacy.load(model_name_trf)
            .RequestException as e:
                st.error(f"Failed to download {filename} from GitHub:st.success(f"Successfully downloaded and loaded '{model_name_trf}'.")
            return nlp
        except Exception as e_trf:
            st.error {e}")
                # Attempt to remove potentially corrupted file
                if os.path(f"Failed to download/load '{model_name_trf}': {e_trf}").exists(local_path):
                    os.remove(local_path
            st.info(f"Attempting to load fallback model '{model_name_sm}'...")
            try:
                nlp = spacy.load(model_)
                return False # Stop further downloads if one fails
    # Check again after attempting downloads
    if not all_files_present:
        for filename in MODEL_FILES:
name_sm)
                st.warning(f"Loaded smaller spaCy model '{model_name_            if not os.path.exists(os.path.join(modelsm}'. NER might be less accurate.")
                return nlp
            except OSError:
                 st.warning(f"SpaCy fallback model '{model_name_sm}'_dir, filename)):
                 st.error(f"Model file {filename} is missing after not found. Attempting to download...")
                 try:
                     spacy.cli.download(model_name_sm)
                     nlp = spacy.load(model_name_sm)
                     st.warning(f"Successfully downloaded and loaded fallback download attempt.")
                 return False # Indicate failure if any file is still missing
    return True

# Load spaCy model for NER
@st.cache_resource
 '{model_name_sm}'. NER might be less accurate.")
                     return nlp
                 except Exception asdef load_spacy_model():
    """Loads the spaCy model, attempting e_sm:
                     st.error(f"Failed to download/load fallback spaCy model '{model_name_sm}': {e_ download if necessary."""
    model_name = "en_core_web_trf"
sm}")
                     return None # Indicate failure to load any spaCy model

    fallback_model_name = "en_core_web_sm"
    # Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading Language Model...")
def load_model_and_tokenizertry:
        # Check if model is installed, if not download
        if not():
    """Loads the GPT-2 model and tokenizer after attempting download."""
    if not download_ spacy.util.is_package(model_name):
             st.info(f"Downloadingmodel_files(MODEL_DIR):
        st.error("Model download failed. Cannot spaCy model '{model_name}'...")
             spacy.cli.download(model_ load Language Model.")
        return None, None
    try:
        name)
             st.info(f"Finished downloading '{model_name}'.")
        nlpmodel = GPT2LMHeadModel.from_pretrained(MODEL_DIR, = spacy.load(model_name)
        st.success(f"Loaded spaCy model '{model_name}'.")
        return nlp
     trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
        st.success("Language Model and Tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer fromexcept (OSError, SystemExit, Exception) as e: # Catch broader {MODEL_DIR}: {e}")
        # Clean up potentially corrupted cache errors during download/load
        st.error(f"Failed to load spaCy model '{model_name}': {e}")
        st.info(f"Attempting to load fallback '{fallback_model_name}'...")
        try:
            if not spacy.util.is_ directory if loading fails
        # Note: This is aggressive, use with caution orpackage(fallback_model_name):
                st.info(f"Downloading add more specific checks
        # if os.path.exists(MODEL_DIR): fallback spaCy model '{fallback_model_name}'...")
                spacy.cli
        #     import shutil
        #     try:
        #         sh.download(fallback_model_name)
                st.info(futil.rmtree(MODEL_DIR)
        #         st.warning"Finished downloading '{fallback_model_name}'.")
            nlp = spacy.load(fallback_model_name)
            st.warning(f"Loaded smaller("Cleared potentially corrupted model cache. Please refresh.")
        #     except Exception fallback spaCy model '{fallback_model_name}'. NER might be less accurate as clean_e:
        #         st.warning(f"Could.")
            return nlp
        except (OSError, SystemExit, Exception) as not clear model cache directory: {clean_e}")
        return None, None

 e2:
            st.error(f"Failed to load fallback spaCy model '{fallback_model_name}': {e2}")
            # Define static placeholders
static_placeholders = {
    "{{APP}}": "<b>Appst.error("NER functionality will be limited.")
            return None


# Load the</b>", "{{ASSISTANCE_SECTION}}": "<b>Assistance Section</b>", "{{CANCEL_T DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner="Loading Language Model...")ICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>", "{{CANCEL_TICKET_SECTION
def load_model_and_tokenizer():
    """Downloads (if necessary) and loads the language model and tokenizer."""
    if not download_model_files(MODEL_DIR):
        st.}}": "<b>Ticket Cancellation</b>", "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>",
    "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>", "{{CANCELLerror("Core model files download failed. Cannot load language model.")
        return None,ATION_OPTION}}": "<b>Cancellation</b>", "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{C None
    try:
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR,ANCELLATION_SECTION}}": "<b>Track Cancellation</b>", "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>", "{{CHECK_C trust_remote_code=True)
        tokenizer = GPT2Tokenizer.ANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>from_pretrained(MODEL_DIR)
        st.success("Language model and",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e</b>", "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>",:
        st.error(f"Error loading model/tokenizer from { "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund PolicyMODEL_DIR}: {e}")
        # Clean up potentially corrupted model dir if</b>",
    "{{CONNECT_WITH_ORGANIZER}}": "<b>Connect with Organizer</b>", "{{CONTACT_SECTION}}": "<b>Contact</b>", "{{CONTACT_SUPPORT_LINK}}": "www.support-team loading fails? Maybe too risky.
        return None, None

# Define static placeholders
static_place.com",
    "{{CURRENT_TICKET_DETAILS}}": "<b>Current Ticket Details</b>", "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>", "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}": "<b>Customer Support</b>holders = {
    "{{APP}}": "<b>App</b>", "{{", "{{CUSTOMER_SUPPORT_SECTION}}": "<b>Customer Support</b>", "{{DELIVERY_PERIOD_INFORMATION}}ASSISTANCE_SECTION}}": "<b>Assistance Section</b>", "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_": "<b>Delivery Period</b>",
    "{{DELIVERY_SECTION}}OPTION}}": "<b>Cancel Ticket</b>", "{{CANCEL_TICKET_SECTION}}": "<b>Ticket Cancellation</b>", "{{CANCELLATION_FEE": "<b>Delivery</b>", "{{EDIT_BUTTON}}": "<b>Edit</b>", "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",_INFORMATION}}": "<b>Cancellation Fee Information</b>",
    "{{CANCELLATION_
    "{{EVENTS_PAGE}}": "<b>Events</b>", "{{EVENTSFEE_SECTION}}": "<b>Cancellation Fee</b>", "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>", "{{CANCELLATION__SECTION}}": "<b>Events</b>", "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMINGPOLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CANCELLATION_SECTION}}": "<b>Track Cancellation</b>", "{{CHECK_CANCELLATION_FEE_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>", "{{GET_REFUND_BUTTON}}": "<b>Get Refund</b>", "{{GET_REF_INFORMATION}}": "<b>Check Cancellation Fee Information</b>", "{{CHECK_CUND_OPTION}}": "<b>Get Refund</b>",
    "{{HELP_SECTION}}": "<b>Help</b>", "{{PAYMENT_ISSUE_ANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>OPTION}}": "<b>Payment Issue</b>", "{{PAYMENT_METHOD}}": "<b>Payment</b>",
    "{{PAYMENT_OPTION}}": "",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>", "{{CHECK_PRIVACY_POLICY_<b>Payment</b>", "{{PAYMENT_SECTION}}": "<b>Payments</b>",OPTION}}": "<b>Check Privacy Policy</b>", "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>",
    "{{CONNECT_WITH_ORGANIZER}}": "<b>Connect with Organizer</b>", "{{ "{{PAYMENTS_HELP_SECTION}}": "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}": "<b>Payments</b>", "{{PRIVCONTACT_SECTION}}": "<b>Contact</b>", "{{CONTACT_SUPPORT_LINKACY_POLICY_LINK}}": "<b>Privacy Policy</b>", "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>}}": "www.support-team.com",
    "{{CURRENT_TICKET_DETAILS}}": "<b>Current Ticket Details</b>", "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>", "{{CUSTOMER_SUPPORT", "{{REFUND_SECTION}}": "<b>Refund</b>", "{{REFUND_PAGE}}": "<b>Customer Support</b>",
    "{{CUSTOMER_SUPPORT_PORT_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{REPORT_PAYMENT_AL}}": "<b>Customer Support</b>", "{{CUSTOMER_SUPPORT_SECTION}}": "<b>Customer Support</b>",PROBLEM}}": "<b>Report Payment</b>", "{{SAVE_BUTTON}}": "<b>Save</b> "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{DELIVERY_SECTION}}": "<b>Delivery</b>", "{{EDIT", "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SELL_T_BUTTON}}": "<b>Edit</b>", "{{EVENT_ORGANIZER_OPTIONICKET_OPTION}}": "<b>Sell Ticket</b>", "{{SEND_BUTTON}}": "<b>Send</b>", "{{}}": "<b>Event Organizer</b>",
    "{{EVENTS_PAGE}}SUPPORT_ SECTION}}": "<b>Support</b>", # Note space in original key
    "{{SUPPORT_CONTACT_LINK}}": "www.support-team.com", "{{": "<b>Events</b>", "{{EVENTS_SECTION}}": "<b>Events</b>", "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>", "{{GET_REFUND_BUTTON}}": "<b>Get Refund</b>", "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    "{{HELP_SECTION}}": "<b>Help</b>", "{{PAYMENT_ISSUE_OPTION}}": "<b>Payment Issue</b>", "{{PAYMENT_METHOD}}": "<b>PaymentSUPPORT_SECTION}}": "<b>Support</b>", "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{TICKET_AVAILABILITY_TAB}}": "<b>Ticket Availability</b>", "{{TICKET_DETAILS}}": "<b>Ticket Details</b>", "{{TICKET_INFORMATION}}": "<b>Ticket Information</b>",
    "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>", "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>", "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>", "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>", "{{TICKET_TRANSFER_TAB}}": "</b>",
    "{{PAYMENT_OPTION}}": "<b>Payment</b>", "{{PAYMENT_SECTION}}": "<b>Payments</b>", "{{PAYMENTS_HELP_SECTION}}": "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}": "<b>Payments</b>", "{{PRIV<b>Ticket Transfer</b>",
    "{{TICKETING_PAGE}}": "<b>Ticketing</b>", "{{TICKETS_TAB}}": "<b>Tickets</b>", "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TRANSFER_TICKET_OPTION}}ACY_POLICY_LINK}}": "<b>Privacy Policy</b>", "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>", "{{REFUND_SECTION}}": "<b>Refund</b>", "{{REFUND": "<b>Transfer Ticket</b>", "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>", "{{UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>", "{{UPGRADE_TICKET_INFORMATION}}": "<b>Ticket Upgradation</b>", "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{VIEW_CANCELL_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{REPORT_PAYMENT_PROBLEM}}": "<b>Report Payment</b>", "{{SAVE_BUTTON}}": "ATION_POLICY}}": "<b>View Cancellation Policy</b>", "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>", "{{WEBSITE_URL}}": "www.events-ticketing.com"
}

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static<b>Save</b>", "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>", "{{SEND_BUTTON}}": "<b>Send</b>", "{{SUPPORT_ SECTION}}": "<b>Support</b>", # Note space in original key
    "{{SUPPORT_CONTACT_LINK}}": "www.support-team.com", "{{SUPPORT_SECTION}}": "<b>Support</b>", "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{TICKET_AVAIL_placeholders):
    """Replaces static and dynamic placeholders in the response text."""
ABILITY_TAB}}": "<b>Ticket Availability</b>", "{{TICKET_DETAILS}}": "<b>Ticket Details</b>", "{{TICKET_INFORMATION}}": "    response = str(response) # Ensure response is a string
    # Replace static placeholders first
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    <b>Ticket Information</b>",
    "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>", "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>", "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{TICKET_SECTION}}# Then replace dynamic placeholders
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return": "<b>Ticketing</b>", "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>", "{{TICKET_TRANSFER_TAB}}": "<b>Ticket Transfer</b>",
    "{{TICKETING_PAGE}}": "<b>Ticketing</b>", response

# Function to extract dynamic placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    """Extracts EVENT and GPE/LOC entities using "{{TICKETS_TAB}}": "<b>Tickets</b>", "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b> spaCy, provides defaults."""
    dynamic_placeholders = {}
    if nlp:", "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>", "{{ # Check if nlp model loaded successfully
        try:
            doc = nlp(user_question)
            for ent in doc.ents:
                if ent.label_ == "EVENT":UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>", "{{UPGRADE_TICKET_INFORMATION}}": "<b>Ticket Upgradation</b>", "{{
                    event_text = ent.text.title() # Capitalize eventUPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>", "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>", "{{WEBSITE_URL}}": "www.events-ticketing.com name
                    dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                elif ent.label_ in ["G"
}

# Function to replace placeholders
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    """RePE", "LOC"]: # Include GPE (Geo-Political Entity) and LOC (Locationplaces static and dynamic placeholders in the response string."""
    response = str(response)
                    city_text = ent.text.title() # Capitalize city) # Ensure response is string
    # Apply static placeholders first
    for placeholder, value in static_placeholders.items():
        response = response./location name
                    dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
        except Exception as e:
            st.warning(f"NER processing failed: {e}") # Warn userreplace(placeholder, value)
    # Apply dynamic placeholders
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Function to extract dynamic about NER issue
    # Provide default values if entities were not found
    if '{{EVENT placeholders using SpaCy
def extract_dynamic_placeholders(user_question, nlp):
    """Extracts EVENT and CITY entities using spaCy,}}' not in dynamic_placeholders:
        dynamic_placeholders[' provides defaults."""
    dynamic_placeholders = {
        '{{EVENT{{EVENT}}'] = "the event" # More natural default
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{}}': "the event", # Default value
        '{{CITY}}': "your city"  # Default value
    }
    if nlp and userCITY}}'] = "your city" # More natural default
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def_question: # Check if nlp model loaded and question exists
        try:
             generate_response(model, tokenizer, instruction, max_length=25doc = nlp(user_question)
            for ent in doc.6):
    """Generates a response using the loaded GPT-2 model."""
    ifents:
                # Prioritize longer matches if multiple overlaps? (spaCy usually handles this)
                if ent.label_ == "EVENT":
                     not model or not tokenizer:
        return "Error: Model or tokenizer not available for response generation."
    try:
        model.eval() # Setevent_text = ent.text.strip().title()
                    dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
                elif ent.label_ in ["GPE", "LOC"]: model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to # Include LOC for more general locations
                    city_text = ent.text(device)

        # Prepare the input text in the format expected by the fine-tuned model
        input_text = f"Instruction: {instruction}.strip().title()
                    # Avoid replacing if a more specific entity was already Response:"
        inputs = tokenizer(input_text, return_tensors="pt", padding found? (e.g., GPE > LOC?)
                    # For simplicity, last=True, truncation=True, max_length=512).to one found wins for now.
                    dynamic_placeholders['{{CITY}}'] = f"<b>(device)

        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model.generate(
                input_ids=inputs["{city_text}</b>"
        except Exception as e:
            input_ids"],
                attention_mask=inputs["attention_mask"],st.warning(f"NER processing failed during extraction: {e}") # Warn
                max_new_tokens=max_length, # Control the length of the *generated* text
                num_return_sequences=1, # Generate only one response
                temperature=0.7, # Controls randomness ( but continue
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256lower = more deterministic)
                top_p=0.95, #):
    """Generates a response using the loaded model and tokenizer."""
    if not model or Nucleus sampling: considers top p% probability mass
                do_sample=True, not tokenizer:
        return "Error: Language Model or Tokenizer is not available." # Enable sampling for less repetitive responses
                pad_token_id=tokenizer.eos_token_id
    try:
        model.eval()
        device = torch.device("cuda" if # Set padding token for generation
            )

        # Decode only the newly torch.cuda.is_available() else "cpu")
        model.to(device)

        # Prepare input text
        input_text = f"Instruction: {instruction} Response:"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
             generated tokens, excluding the input prompt
        response_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(response_tokens, skip_specialtruncation=True,
            max_length=512 # Max_tokens=True)
        return response.strip()

    except Exception as e:
        st.error(f"Error during response generation: length the model can handle for input
        ).to(device)

        # {e}")
        return "Sorry, I encountered an technical difficulty while generating the response."


# --- CSS Styling ---
st.markdown(
    """
<style>
 Generate output
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention/* General Button Styles (Apply to Reset, Continue) */
.stButton>button_mask=inputs["attention_mask"],
                max_new_tokens=max_length, # Max tokens to *generate*
                num_return_ {
    color: white !important;
    border: none;
    border-radius: 25px;
    padding: 10pxsequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id= 20px;
    font-size: 1.1emtokenizer.eos_token_id
            )

        # Decode only the generated part, excluding the prompt
; /* Adjusted size */
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: inline-flex;
    align-items: center;
    justify        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].-content: center;
    margin-top: 10px; /* Increased margin */
    width: auto;
    min-width: 100px;
    font-family: 'Times New Roman', Times, serif !important;
}
.stButton>button:hover {shape[-1]:], skip_special_tokens=True)
        return response_text.strip()

    except Exception as e:
        st.error(f"Error during response generation: {e}")
        # Log the error for debugging
        print(f"Error generating response for instruction
    transform: scale(1.05);
    box-shadow: '{instruction}': {e}")
        return "Sorry, I encountered an technical difficulty while generating the response. Please try asking differently."


# --- CSS Styling ---
 0px 5px 15px rgba(0, 0, 0, 0.3);
    color: white !important;
}
.stButton>button:active {
    transform: scale(0.98st.markdown(
    """
<style>
/* General Button Styles (Reset, Continue) */
.stButton>button {
    color: white !important;
    border: none;
    border-radius: 25px;);
}

/* Style for 'Continue' Button */
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button[kind="secondary"] { /* Adjust selector if needed */
   background: linear-gradient(90
    padding: 10px 20px;
    font-size: 1.1em; /* Adjusted size */
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0deg, #ff8a00, #e52e71) !important; /* Orange/Pink gradient for Continue */
}

/* Style for 'Reset Chat.2s ease;
    display: inline-flex;
    align' Button */
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button[kind="secondaryFormSubmit"] { /* Specific selector for Reset */-items: center;
    justify-content: center;
    margin-top: 10px; /* Increased margin */
    width: auto;
   background: linear-gradient(90deg, #6c757d, #343a40) !important; /* Gray gradient for Reset */
}


/* Style
    min-width: 100px;
    font-family: 'Times New Roman', Times, serif !important;
}
 for 'Ask this question' Button */
div[data-testid="stHorizontal.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 5px 15px rgba(0Block"] div[data-testid="stButton"] button:nth-of-type(1), 0, 0, 0.3);
    color: white !important;
}
.stButton>button:active {
    transform: scale( { /* Original selector for Ask button */
    background: linear-gradient(90deg, #29ABE2, #0077B6) !important; /* Blue gradient */
    color0.98);
}

/* Style for 'Continue' Button */: white !important;
    font-size: 1.1em
/* Selects the button within the specific columns layout used for Continue */
div[data- !important; /* Match other buttons */
    padding: 8px 18px !important; /* Slightly adjust padding if needed */
    margin-top:testid="stHorizontalBlock"] > div:nth-child(2) . 0px !important; /* Align with selectbox */
}

/* Style for the Regenerate Button (Circle) */
div[data-testid="stChatMessage"] div[data-testid="stButton"] button {
    background-color: #f0f2f6 !important;stButton>button { /* Tighter selector */
    background: linear-gradient(90deg,
    color: #333 !important;
    border: 1px solid #ccc !important;
    border-radius: 50% !important;       #ff8a00, #e52e71) !important; /* Orange/Pink gradient */
}


/* Style for 'Reset Chat' Button/* Key for circle */
    padding: 0 !important;             /* Remove padding */
    font-size: 1.0em !important;        /* Icon size */
    font-weight */
/* Assuming it's typically at the bottom, possibly in a vertical block */
div[data-testid="stVerticalBlock"] .stButton>button[kind="secondary"]: normal !important;
    min-width: 30px ! { /* Example selector, might need adjustment */
   background: linear-gradient(90deg, #important;         /* Width = Height */
    width: 30px !important;             /* Width = Height */
    height: 30px !important;6c757d, #343a40) !important            /* Width = Height */
    margin-left: 8px !important;        /* Space from text */
    margin-top: 2px !important;         /* Fine-tune vertical alignment */
    /* Flexbox for perfect centering of; /* Gray gradient for reset */
   /* If Reset Button is not secondary the icon */
    display: inline-flex !important;
    align-items: center !important;     /* Vertical center */
    justify-, adjust selector: */
   /* e.g., div[data-testidcontent: center !important; /* Horizontal center */
    line-height: 1 !important;          /* Reset line-height for flex */
    /* Override other potential styles */
    background: #f0f2f6 !important;
="stVerticalBlock"] .stButton>button:nth-child(X    box-shadow: none !important;        /* Clear default shadow */
    /* Ensure) */
}
/* Or target specifically by key if stable */
. transitions apply */
    transition: transform 0.2s ease, box-shadow stButton button[data-testid="stButton-reset_button"] { /*0.2s ease, background-color 0.2s ease ! Using data-testid from key */
     background: linear-gradient(90deg, #6c757d, #343a40) !importantimportant;
}
div[data-testid="stChatMessage"] div[data-testid="stButton"] button:hover {
    background-color: #e0e2e6 !important;
    color: #; /* Gray gradient */
}

/* Style for 'Ask this question' Button */000 !important;
    transform: scale(1.1) !important;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2
/* Uses the horizontal block layout for the dropdown and button */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {) !important; /* Keep hover shadow */
}
div[data-testid="stChatMessage"] div[data-testid="stButton"] button:active {
    transform: scale(1.0) !important;
}


/* Apply Times New Roman to all text elements */
* {
    font-family:
    background: linear-gradient(90deg, #29ABE2, #0077B6) !important; /* Blue gradient */
    color: white !important;
    font-size: 1.0em ! 'Times New Roman', Times, serif !important;
}

/* Specific adjustments for Streamlit elements */
.stSelectbox > div > div >important; /* Slightly smaller */
    padding: 8px 15 div > div { font-family: 'Times New Roman', Times, serifpx !important;
    margin-top: 0px !important; /* Align with dropdown */
    margin-left: 5px;
}


/* Style for the !important; }
.stTextInput > div > div > input { font-family: 'Times New Roman', Times, serif !important; }
.stTextArea > div > div > textarea { font-family: 'Times Regenerate Button (Circle) */
div[data-testid="stChatMessage"] div[data-testid="stButton"] button {
    background-color New Roman', Times, serif !important; }
.stChatMessage {
: #f0f2f6 !important;
    color: #333 !important;
    border: 1px solid #ccc !    font-family: 'Times New Roman', Times, serif !important;important;
    border-radius: 50% !important;      
    line-height: 1.6; /* Improve readability */
}
.st-/* Key for circle */
    padding: 0 !important;             /* Remove padding */
    font-emotion-cache-r421ms { font-family: 'Times Newsize: 1.0em !important;        /* Icon size */
    font-weight: Roman', Times, serif !important; } /* For warnings/errors */
.streamlit-expanderContent { font-family: 'Times New Roman', Times, serif !important; }

/* Horizontal Line Separator */
.horizontal-line {
    border-top: 1px solid #ccc; /* Thinner, normal !important;
    min-width: 30px !important;         /* Width = Height */
    width: 30px !important;             /* Width = Height */
    height: 30px !important;            /* Width = Height */
    margin-left: 8px !important;
    margin-top: 0px !important;         /* Align vertically lighter line */
    margin: 20px 0; /* More with text line */
    /* Flexbox for perfect centering of the icon */
    display: inline vertical space */
}

/* Chat Input Shadow Effect */
div[data-testid="stChatInput-flex !important;
    align-items: center !important;     /* Vertical center */
    justify-content: center !important; /* Horizontal center */
    line-height: 1 !important;          /* Reset line-height for flex"] {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Softer shadow */
    border-radius: 10px; /* More rounded */
    padding: 10px 15px;
    margin: 15px  */
    /* Override other potential styles */
    background: #f00;
    background-color: #ffffff; /* Ensure white background */
}
</style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit UI ---
st.markdown("<h1 style='font-size: 43px; text-align: center;f2f6 !important;     /* Ensure background override */
    box-shadow: none !important;        /* Clear default shadow */
}
div[data-testid color: #333;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)
st.markdown("---") # Add a="stChatMessage"] div[data-testid="stButton"] button:hover separator line

# Initialize session state variables
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "models_loaded" not in st.session {
    background-color: #e0e2e6 !important_state:
    st.session_state.models_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "regener;
    color: #000 !important;
    transform: scale(1.1) !ate_index" not in st.session_state:
    st.important;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2) !important; /* Keep hover shadow */
}
div[data-testid="stChatMessage"] div[data-testid="stButton"] button:active {
    transform: scale(1.0) !important;
}


/* Apply Times New Roman to all text elements */
* {
    font-family: 'Times New Roman', Times,session_state.regenerate_index = None
if "nlp" not in st.session_state: # Explicitly track models in state
    st.session_state.nlp = None
if "model" not in st.session_state serif !important;
}
/* Specific adjustments for Streamlit elements */
.stSelectbox > div >:
    st.session_state.model = None
if "tokenizer div > div > div { font-family: 'Times New Roman', Times" not in st.session_state:
    st.session_state.tokenizer = None


# Example queries for dropdown
example_queries = [
    "How do I buy a ticket?",
    "How can I upgrade my ticket, serif !important; }
.stTextInput > div > div > input { font-family: 'Times New Roman', Times, serif !important; }
.stTextArea > div > div > textarea { font-family: for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?",
    "How can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?", 'Times New Roman', Times, serif !important; }
.stChatMessage
    "What is the ticket cancellation fee?",
    "How can I { font-family: 'Times New Roman', Times, serif !important; }
.stMarkdown { font-family: 'Times New Roman', Times, serif !important; } track my ticket cancellation status?",
    "How can I sell my ticket?" /* Ensure markdown uses it */
.stAlert { font-family: 'Times New Roman
]

# --- Model Loading Logic ---
# This block runs only once or until models are loaded successfully
if not st.session_state.models_', Times, serif !important; } /* Errors, warnings */
.streamlit-expanderHeader { font-family: 'Times New Roman', Times, serif !important; }
.streamlit-expanderContent { font-family: 'Times New Roman',loaded:
    with st.spinner("Initializing chatbot... Loading models and resources... Please wait..."): Times, serif !important; }

/* Horizontal Line */
.horizontal-line {
    
        nlp_model = load_spacy_model()
        llm_model, llm_tokenizer = load_model_and_tokenizer()

        ifborder-top: 2px solid #eee; /* Lighter line */
    margin: 20 llm_model is not None and llm_tokenizer is not None:
            st.session_state.model = llm_model
            st.session_state.tokenizer = llm_tokenizer
            st.session_state.nlp = npx 0; /* More spacing */
}

/* Chat Input Shadow */
div[data-testid="stChatInput"] {
    box-shadow:lp_model # Store even if None (indicates NER failure)
            st.session_ 0 4px 8px rgba(0, 0, state.models_loaded = True
            st.rerun() # R0, 0.1); /* Softer shadow */
    border-radius: 8px; /* Slightly less rounded */
    padding: 10px 15px;erun after successful load to proceed
        else:
            # Errors are already displayed by /* Adjust padding */
    margin: 15px 0;
    background the loading functions
            st.stop() # Stop execution if core models failed-color: #ffffff;
}
</style>
    """,
    unsafe_allow_html=True,
)


# --- Streamlit UI ---
st


# --- Disclaimer Logic ---
# Show disclaimer only after models are loaded but before chat starts
if st.session_state.models_loaded and not st..markdown("<h1 style='font-size: 43px; font-family: Times New Roman, Times, serif;'>Advanced Events Ticketing Chatbot</h1>", unsafe_allow_html=True)

# --- Session State Initialization ---
#session_state.show_chat:
    st.markdown(
         Using .setdefault() is a concise way to initialize if not present
st.session_state.setdefault('"""
        <div style="background-color: #fffbe6; padding: 20px;show_chat', False)
st.session_state.setdefault('models border-radius: 10px; color: #8a6d3b; border: 1px solid #ffeeba; font-family: 'Times New Roman',_loaded', False)
st.session_state.setdefault('chat_history', [])
st.session_state.setdefault('regenerate_index Times, serif;">
            <h2 style="font-size: 2', None)
st.session_state.setdefault('nlp', None)
st.session_state.setdefault('model', None)
st.8px; color: #8a6d3b; font-weight: bold; text-align:session_state.setdefault('tokenizer', None)

# Example queries for dropdown
example_queries = [
    "How do I buy a ticket?", center;">⚠️ Disclaimer</h2>
            <p style="font-size: 16px;
    "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?",
    "How line-height: 1.6;">
                This <b>Chatbot</b> is designed for assisting with ticketing inquiries. Due to computational limits, it's fine-tuned on can I find details about upcoming events?",
    "How do I contact customer service?",
    "How do I get a refund?",
    "What is the ticket cancellation fee specific intents and may not answer all queries accurately.
            </p>
            <p style="font-size: 16px; line-height: 1.6?",
    "How can I track my ticket cancellation status?",
    ";">
                Optimized intents include:
            </p>
            <ul style="font-size: 15px; line-height: 1.7; columnsHow can I sell my ticket?"
]

# --- Model Loading Logic ---
# Load models only once when the app starts and models_loaded is False
if not st.session_state.models_loaded:
    with st.spinner("Initializing chatbot components... Please wait..."):
        nlp_: 2; -webkit-columns: 2; -moz-columns: 2;">
                model = load_spacy_model()
        lang_model, tokenizer_model = load<li>Cancel Ticket</li> <li>Buy Ticket</li> <li>Sell Ticket</li> <li>_model_and_tokenizer()

        # Check if loading was successful
        if lang_model is not None and tokenizer_model is not None:
            st.session_state.modelTransfer Ticket</li>
                <li>Upgrade Ticket</li> <li>Find Ticket</li> <li>Change Ticket Details</li>
                <li>Get Refund</li> <li>Find Events</li> <li>Customer = lang_model
            st.session_state.tokenizer = tokenizer_model
 Service</li>
                <li>Check Cancellation Fee</li> <li>Track Cancellation</li> <li>Ticket            st.session_state.nlp = nlp_model # Can be None if spaCy failed
            st.session_state.models_loaded = True
 Information</li>
            </ul>
            <p style="font-size:             st.rerun() # Rerun to proceed to disclaimer or chat
        else:
            st16px; line-height: 1.6;">
                Queries outside these areas might not receive accurate responses. If the chatbot struggles even with listed.error("Essential language model components failed to load. Chatbot cannot start.")
            # Keep models_loaded as False, app will stop here.

# --- Disclaimer Logic ---
# intents, please be patient and try rephrasing your question.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Continue button aligned to the right
    _, col_btn = st.columns([4 Show disclaimer only if models are loaded but chat hasn't been shown yet
if st.session_state.models_loaded and not st.session_state, 1]) # Use columns for alignment
    with col_btn:.show_chat:
    st.markdown(
        """
        <div style="background-color: #fff3cd; padding: 20px;
        if st.button("Continue", key="continue_button"):
            st.session_state. border-radius: 10px; color: #8564show_chat = True
            st.rerun()


# --- Chat04; border: 1px solid #ffeeba; font-family: 'Times New Roman', Times, serif;">
            <h2 style="color: #856404; font Interface Logic ---
# This section runs only if models are loaded and user clicked "Continue"
if st.session_state.models_loaded and st.session_state.show_chat:
    # Access loaded models/resources directly-weight: bold; text-align: center;">⚠️ Disclaimer</h2>
            <p style="font from session state
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session-size: 16px; line-height: 1.6_state.tokenizer

    # --- Handle Regeneration Request ---
    idx_to_regenerate = st.session_state.get('regenerate_index', None)
    if idx_to_regenerate is not None:
        st.session_state.regenerate_;">
                This <b>Chatbot</b> assists with ticketing inquiries based on a specific set of trained intents. Due to computational limits during training, it may not accuratelyindex = None # Reset trigger immediately

        # Validate index and ensure models are available
        if 0 < idx_to_regenerate < len(st.session_state.chat_history) and \
           st.session_state.chat respond to all query types.
            </p>
            <p style="font-size: 16px; line-height: 1_history[idx_to_regenerate - 1]["role"] == "user" and \
           model and tokenizer:

            original_prompt = st.session_state.chat_history[idx_to_regenerate - 1]["content"]
            # Show spinner during regeneration
            with st..6;">
                Optimized intents include:
            </p>
spinner("🔄 Regenerating response..."):
                dynamic_placeholders = extract_dynamic_placeholders(original_prompt, nlp)
                new            <ul style="font-size: 16px; line-_response_gpt = generate_response(model, tokenizer, original_prompt)
                new_full_response = replace_placeholders(new_response_gpt, dynamic_height: 1.6; list-style-position: inside; padding-left: 0;">
                <li>Buying, Selling, Transferring, Upgrading Ticketsplaceholders, static_placeholders)

                # Update the history and rerun</li>
                <li>Finding Tickets & Upcoming Events</li>
                <li>Changing Ticket Details</li>
                <li>Refund
                st.session_state.chat_history[idx_to_regenerate]["content"] = new_full_response
                st.rers & Cancellation Fees</li>
                <li>Tracking Cancellations</li>
                <li>General Ticket Information & Customer Service</li>
            </ul>
            <p style="font-size: 1un()
        else:
            # Handle invalid index or missing models during6px; line-height: 1.6;">
                Queries outside these areas regeneration attempt
            if not model or not tokenizer:
                 st.warning("Cannot regenerate: Models not available.")
            else:
                 st.warning(f"Could not regenerate message at index {idx_to_regenerate}.") # might not receive accurate answers. If the chatbot struggles even within these topics, please rephrase or try again. Your patience is appreciated!
            </p>
        </div> Should generally not happen if logic is correct

    # --- Top Section: Example Queries ---
    st.markdown("#### Quick Start Examples:")
    col_select, col_ask
        """,
        unsafe_allow_html=True
    )

_btn = st.columns([4, 1]) # Columns for selectbox and button
    with col_select:
        selected_query = st.selectbox(
            "Choose a query from examples:",
            ["Choose your question"]    # Continue button aligned to the right using columns
    _, col2 = st.columns([3 + example_queries,
            key="query_selectbox",
            label, 1]) # Adjust ratio if needed for alignment
    with col2:
        if st.button("Continue", key="continue_button"):
            st.session_state._visibility="collapsed" # Hide label to save space
        )
    with col_ask_btn:
        process_query_button = st.button("Ask",show_chat = True
            st.rerun() # Rerun to display the chat interface

# --- Chat Interface Logic ---
# Show chat only if models are loaded AND the user clicked "Continue"
if st.session_state.models_loaded and st.session_ key="query_button", help="Ask the selected example question") # Changed text slightly

    # --- Process Selected Query ---
    if process_query_buttonstate.show_chat:
    st.write("Ask me about ticket cancellations, refunds, or any event-related inquiries!")

    # --- Regeneration Handling ---
    # Check:
        if selected_query == "Choose your question":
            st. if a regeneration was requested in the *previous* run
    idx_to_regenerate = st.session_state.get('regenerate_indextoast("💡 Please select a question from the dropdown first.", icon="⚠️")
        elif selected_query:') # Use .get for safety
    if idx_to_regenerate is not None:
        
            prompt_to_process = selected_query[0].upper() + selected_query# Reset trigger immediately to prevent loops
        st.session_state.regenerate_index = None

        [1:] # Capitalize

            # Append user message
            st.session_# Validate index and context before regenerating
        if 0 < idx_to_regeneratestate.chat_history.append({"role": "user", "content": prompt_to_process, "avatar": "👤"})

            # Generate < len(st.session_state.chat_history) and \
 and append assistant message (show spinner inline)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    if           st.session_state.chat_history[idx_to_regener model and tokenizer: # Check models again before generation
                        dynamic_placeholders = extract_dynamic_placeholders(prompt_to_process, nlp)
ate - 1]["role"] == "user":

            original_prompt                        response_gpt = generate_response(model, tokenizer, prompt_to_process)
                        full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
                        st.session_state.chat_ = st.session_state.chat_history[idx_to_regenerate - 1]["content"]

            # Use cached models from session state
            nlp = st.history.append({"role": "assistant", "content": full_response,session_state.nlp
            model = st.session_state.model
            tokenizer = st.session_state.tokenizer

            # Show spinner during "avatar": "🤖"})
                        st.rerun() # Rerun regeneration
            with st.spinner("🔄 Regenerating response..."):
                 to display the updated history
                    else:
                        st.error("Cannotif model and tokenizer: # Check models again just in case
                    dynamic_placeholders = extract_dynamic_placeholders(original_prompt, nlp) # generate response: Models unavailable.")
                        st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I cannot generate a response right now.", "avatar": "🤖"}) nlp might be None
                    new_response_gpt = generate_response(model, tokenizer, original_prompt)
                    new_full_response = replace_placeholders(new_response_gpt, dynamic_placeholders, static_placeholders)
                        st.rerun()


    st.markdown("---") # Separator

                    # Update the specific message content in the history list
                    st.session_state.chat_history[idx_to_regenerate]["content"] = new_ before chat history

    # --- Display Chat History ---
    st.markdown("#### Conversation History")
    last_role = None # Track role for adding separators
    if not st.session_state.chat_history:
         st.infofull_response
                    st.rerun() # Rerun Streamlit to display the updated("Ask a question using the examples above or type your own below!")

    for idx, message in enumerate(st.session_state.chat_history):
        # Add separator line between different user history
                else:
                    st.error("Cannot regenerate response: Model components not available.")
                    # Optionally, could revert the regenerate_index if needed, but/assistant turns
        if message["role"] == "user" and last_role == "assistant":
            st.markdown("<div class='horizontal reset is safer.
        else:
             # This case might happen if history changes between runs (e.g., reset)
             st.warning(f"Could not regenerate message at index {idx_to_regenerate}. Context might have changed.")
-line'></div>", unsafe_allow_html=True)

        with             # Ensure index is cleared if validation fails
             st.session_state.regenerate_index = None


    # --- Example Query Dropdown and Button ---
    col st.chat_message(message["role"], avatar=message["avatar"]):1, col2 = st.columns([4, 1]) # Adjust ratio for layout
    with col1:
        selected_query = st.selectbox(
            # Display the message content
            st.markdown(message["content"], unsafe_allow_
            "Choose a query from examples:",
            ["Choose your question"] + example_queries,
html=True)

            # Add regenerate button for assistant messages
            if message["role"] == "assistant" and idx > 0 and st.session_state.chat_history[idx - 1]["role"] == "user":
                button_key = f"regenerate_{            index=0, # Default to placeholder
            key="query_selectboxidx}"
                if st.button("🔄", key=button_key, help",
            label_visibility="collapsed" # Hide the label "Choose a query..."="Regenerate this response"):
                    st.session_state.regener
        )
    with col2:
        # Button click handled belowate_index = idx # Set index for next run
                    st.rerun()
        process_query_button = st.button("Ask this", key="query_button") # Trigger rerun to handle regeneration

        last_role = message["role"]

    st.markdown("---") # Separator before chat input

    # ---


    # --- Display Chat History ---
    last_role = None # Track previous message role for visual separation
    for idx, message in enumerate(st.session_state. Chat Input Box ---
    if prompt := st.chat_input("Enter your own question herechat_history):
        is_user = message["role"] == "user..."):
        prompt = prompt.strip()
        if not prompt:
            st.toast"
        # Add separator line only between turns (Assistant then User)
        if is_("⚠️ Please enter a question.", icon="✏️")
        else:
            prompt_to_process = prompt[0].upper() + prompt[1:] # Capitalize

            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": prompt_to_process, "avatar": "👤"})

            # Generate and appenduser and last_role == "assistant":
            st.markdown("<div class='horizontal-line'> assistant message (show spinner inline)
            with st.chat_message("assistant", avatar="🤖"):
                 with st.spinner("Thinking..."):
                    if model and tokenizer</div>", unsafe_allow_html=True)

        with st.chat_message(message["role"], avatar=message["avatar"]):
            # Display the message content using: # Check models
                        dynamic_placeholders = extract_dynamic_placeholders(prompt_ Markdown
            st.markdown(message["content"], unsafe_allow_html=True)

            # Add regenerate button ONLY for assistant messages that follow a userto_process, nlp)
                        response_gpt = generate_response(model, tokenizer, prompt_to_process)
                        full_response = replace_placeholders(response_gpt, dynamic_placeholders, static message
            if message["role"] == "assistant" and idx > 0 and st.session_state.chat_history[idx - 1]["role"]_placeholders)
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar == "user":
                button_key = f"regenerate_{idx}"
                # Render the button. If clicked, it sets state and triggers rerun.
                if st": "🤖"})
                        st.rerun() # Rerun to display
                    else:
                        st.error("Cannot generate response: Models unavailable.")
                        st.session_.button("🔄", key=button_key, help="Regenerate this response"):
                    # Set the index to regenerate in session state
                    st.session_state.regenerate_index = idx
                    st.rerun() # Rerun to process the regeneration request

        last_role = message["role"] # Update last role


state.chat_history.append({"role": "assistant", "content": "Sorry, I cannot generate a response right now.", "avatar": "🤖"})
                        st.rerun()

    # --- Reset Button ---
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True) #    # --- Process Actions (Dropdown Ask or Chat Input) ---

    # Function to handle Add some space before reset button
        if st.button("Reset Chat", key="reset_button"):
            st.session_state.chat_history = []
             response generation and history update
    def handle_user_query(prompt):st.session_state.regenerate_index = None # Clear regeneration state
            st.success("Chat history cleared.")
            time.sleep(0.5) # Brief
        """Adds user query, generates response, adds to history, and rer pause before rerun
            st.rerun()
