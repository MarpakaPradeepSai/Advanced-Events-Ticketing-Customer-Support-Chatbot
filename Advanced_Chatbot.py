import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import spacy
import time
import base64

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
            response = requests.get(url)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
            else:
                st.error(f"Failed to download {filename} from GitHub.")
                return False
    return True

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

# Load the DistilGPT2 model and tokenizer
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model_dir = "/tmp/DistilGPT2_Model"
    if not download_model_files(model_dir):
        st.error("Model download failed. Check your internet connection or GitHub URL.")
        return None, None

    model = GPT2LMHeadModel.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Define static placeholders
static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{CONTACT_SUPPORT_LINK}}": "www.support-team.com",
    "{{SUPPORT_CONTACT_LINK}}": "www.support-team.com",
    "{{CANCEL_TICKET_SECTION}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    "{{UPGRADE_TICKET_INFORMATION}}": "<b>Upgrade Ticket Information</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>",
    "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>",
    "{{APP}}": "<b>App</b>",
    "{{CHECK_CANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>",
    "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>",
    "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>",
    "{{SAVE_BUTTON}}": "<b>Save</b>",
    "{{EDIT_BUTTON}}": "<b>Edit</b>",
    "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>",
    "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>",
    "{{PRIVACY_POLICY_LINK}}": "<b>Privacy Policy</b>",
    "{{REFUND_SECTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>",
    "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>",
    "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",
    "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>",
    "{{CONTACT_SECTION}}": "<b>Contact</b>",
    "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SUPPORT_SECTION}}": "<b>Support</b>",
    "{{EVENTS_SECTION}}": "<b>Events</b>",
    "{{EVENTS_PAGE}}": "<b>Events</b>",
    "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>",
    "{{PAYMENT_SECTION}}": "<b>Payment</b>",
    "{{PAYMENT_OPTION}}": "<b>Payment</b>",
    "{{CANCELLATION_SECTION}}": "<b>Cancellation</b>",
    "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>",
    "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>",
    "{{REFUND_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{DELIVERY_SECTION}}": "<b>Delivery</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>",
    "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>",
    "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>",
    "{{PAYMENT_METHOD}}": "<b>Payment</b>",
    "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>",
    "{{SUPPORT_ SECTION}}": "<b>Support</b>",
    "{{CUSTOMER_SUPPORT_SECTION}}": "<b>Customer Support</b>",
    "{{HELP_SECTION}}": "<b>Help</b>",
    "{{TICKET_INFORMATION}}": "<b>Ticket Information</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>",
    "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_BUTTON}}": "<b>Get Refund</b>",
    "{{PAYMENTS_HELP_SECTION}}": "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}": "<b>Payments</b>",
    "{{TICKET_DETAILS}}": "<b>Ticket Details</b>",
    "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>",
    "{{REPORT_PAYMENT_PROBLEM}}": "<b>Report Payment</b>",
    "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{SEND_BUTTON}}": "<b>Send</b>",
    "{{PAYMENT_ISSUE_OPTION}}": "<b>Payment Issue</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}": "<b>Customer Support</b>",
    "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{TICKET_AVAILABILITY_TAB}}": "<b>Ticket Availability</b>",
    "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>",
    "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>",
    "{{TICKETING_PAGE}}": "<b>Ticketing</b>",
    "{{TICKET_TRANSFER_TAB}}": "<b>Ticket Transfer</b>",
    "{{CURRENT_TICKET_DETAILS}}": "<b>Current Ticket Details</b>",
    "{{UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{CONNECT_WITH_ORGANIZER}}": "<b>Connect with Organizer</b>",
    "{{TICKETS_TAB}}": "<b>Tickets</b>",
    "{{ASSISTANCE_SECTION}}": "<b>Assistance Section</b>",
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
        elif ent.label_ == "GPE":
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"
    return dynamic_placeholders

# Generate a chatbot response using DistilGPT2
def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
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
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# Function to convert image to base64
def get_base64_image():
    # For this example, I'll use a simple robot emoji as a placeholder
    # In practice, you would replace this with your actual bot image file
    # Here's how you would do it with a local file:
    # with open("path_to_your_image.png", "rb") as image_file:
    #     encoded = base64.b64encode(image_file.read()).decode()
    
    # Using a simple robot emoji as base64 for demonstration
    # You can replace this with your own image's base64 string
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAA8CAYAAAB6/NlyAAAACXBIWXMAAAsTAAALEwEAmpwYAAADcElEQVR4nO2ZTWgTQRTHf4xCoaAoFQoK8uIloV0o6CWEiIhYUcRLEC9evCjeoqJ4sVJ06cWLXryoV0VBQUEQvHgRFAvFgoKCFw+ChYKCgqD8eO/MnJk5szuzO/PfzvzmzJk5MzP7zswkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZIkSZLkb8lX8DvwG/Ao8CzwKPAwcA/wOvA5cKBL+4HngG8BPwMHgFeBPwMHgFeBPwPngF8DHwT+DDwLPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8CjwKPAo8C
