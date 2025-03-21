import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import requests
import zipfile

# Function to download model files from GitHub
def download_model_files():
    model_dir = "DistilGPT2_Model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        url = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"
        st.write("Downloading model files from GitHub...")
        
        # Assuming the model files are in a directory; adjust if it's a zip or individual files
        # Here, we'll assume they're individual files or a zip (update URL as needed)
        files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            # Add other files if necessary
        ]
        
        for file_name in files:
            file_url = f"{url}/{file_name}"
            response = requests.get(file_url)
            if response.status_code == 200:
                with open(os.path.join(model_dir, file_name), 'wb') as f:
                    f.write(response.content)
            else:
                st.error(f"Failed to download {file_name}")
        st.write("Model files downloaded successfully!")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_dir = "DistilGPT2_Model"
    download_model_files()
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Generate response function
def generate_response(model, tokenizer, device, instruction, max_length=256):
    model.eval()
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
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

# Streamlit app
def main():
    st.title("Advanced Events Ticketing Chatbot")
    st.write("Ask me anything about event ticketing!")

    # Load model
    model, tokenizer, device = load_model()

    # User input
    user_input = st.text_input("Your question:", "How to cancel my ticket?")
    
    if st.button("Get Response"):
        if user_input:
            # Capitalize first letter
            user_input = user_input[0].upper() + user_input[1:]
            with st.spinner("Generating response..."):
                response = generate_response(model, tokenizer, device, user_input)
            st.write("**Chatbot Response:**")
            st.write(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
