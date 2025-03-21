import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import spacy
import os

# Cache the spaCy model (for NER)
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_trf")
    return nlp

# Load the GPT-2 model and tokenizer from your GitHub directory
@st.cache_resource
def load_gpt2_model():
    model_path = "DistilGPT2_Model"  # This is the directory inside your GitHub repository
    model_url = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"
    
    # Use `from_pretrained` with URL to directly load the model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_url)
    tokenizer = GPT2Tokenizer.from_pretrained(model_url)
    
    return model, tokenizer

# Define static placeholders (same as in your previous code)
static_placeholders = {
    "{{WEBSITE_URL}}": "www.events-ticketing.com",
    "{{SUPPORT_TEAM_LINK}}": "www.support-team.com",
    "{{CONTACT_SUPPORT_LINK}}": "www.support-team.com",
    "{{SUPPORT_CONTACT_LINK}}": "www.support-team.com",
    "{{CANCEL_TICKET_SECTION}}": "<b>Cancel Ticket</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    # Add more static placeholders as needed
}

# Extract dynamic placeholders using spaCy NER
def extract_dynamic_placeholders(instruction, nlp):
    doc = nlp(instruction)
    dynamic_placeholders = {}

    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE":  # GPE for cities
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"

    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"

    return dynamic_placeholders

# Replace both static and dynamic placeholders in the response
def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

# Generate the response from the GPT-2 model
def generate_response(instruction, model, tokenizer, nlp, max_length=256):
    dynamic_placeholders = extract_dynamic_placeholders(instruction, nlp)
    input_text = f"Instruction: {instruction} Response:"

    # Tokenize the input and run the model
    inputs = tokenizer(input_text, return_tensors='pt')
    device = model.device
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the response part after "Response:"
    response_start = response.find("Response:") + len("Response:")
    raw_response = response[response_start:].strip()

    # Replace the placeholders
    final_response = replace_placeholders(raw_response, dynamic_placeholders, static_placeholders)

    return final_response

# Streamlit UI code
def main():
    st.title("Event Ticketing Chatbot")
    
    # Load the models
    nlp = load_spacy_model()
    model, tokenizer = load_gpt2_model()

    # Get user input
    user_question = st.text_input("Ask a question about event ticketing:")

    if user_question:
        user_question = user_question[0].upper() + user_question[1:]  # Capitalize first letter
        
        # Generate response
        response = generate_response(user_question, model, tokenizer, nlp)
        
        # Display the response
        st.write(f"**User**: {user_question}")
        st.write(f"**Chatbot**: {response}")

if __name__ == "__main__":
    main()
