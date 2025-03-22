# app.py
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
from safetensors.torch import load_file

# Configure environment for safetensors
os.environ["SAFETENSORS_LAZY_LOAD"] = "1"

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        # Verify safetensors file integrity
        with load_file("DistilGPT2_Model/model.safetensors") as f:
            if not f.keys():
                st.error("Invalid or corrupted safetensors file")
                return None, None

        # Load model with safetensors
        model = GPT2LMHeadModel.from_pretrained(
            "DistilGPT2_Model",
            use_safetensors=True,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Load tokenizer from the directory
        tokenizer = GPT2Tokenizer.from_pretrained(
            "DistilGPT2_Model",  # Path to the directory containing tokenizer files
            local_files_only=True
        )

        # Configure padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Generate response from the model
def generate_response(instruction, model, tokenizer, max_length=256):
    try:
        device = model.device
        model.eval()

        # Format input text
        input_text = f"Instruction: {instruction} Response:"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

        # Generate response
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

        # Decode and extract response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = response.find("Response:") + len("Response:")
        return response[response_start:].strip()

    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit app
def main():
    st.title("Advanced Events Ticketing Chatbot 🤖")
    st.write("Welcome! Ask me anything about ticket booking, events, or cancellations.")

    # Load model with progress
    with st.spinner("Loading AI model..."):
        model, tokenizer = load_model()

    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please check the model files and try again.")
        return

    # User input
    user_input = st.text_input("Ask your question here:")

    if user_input:
        # Generate response with loading indicator
        with st.spinner("Generating answer..."):
            formatted_input = user_input[0].upper() + user_input[1:]
            response = generate_response(formatted_input, model, tokenizer)

        # Display response
        st.subheader("Answer:")
        st.write(response)

if __name__ == "__main__":
    main()
