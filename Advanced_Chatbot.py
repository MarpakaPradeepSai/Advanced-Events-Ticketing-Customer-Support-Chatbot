# app.py
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

@st.cache_resource
def load_model():
    # Load model and tokenizer from local directory
    model = GPT2LMHeadModel.from_pretrained('DistilGPT2_Model')
    tokenizer = GPT2Tokenizer.from_pretrained('DistilGPT2_Model')
    return model, tokenizer

def generate_response(instruction, model, tokenizer, max_length=256):
    device = model.device
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

def main():
    st.title("Advanced Events Ticketing Chatbot 🤖")
    st.write("Welcome! Ask me anything about ticket booking, events, or cancellations.")
    
    # Load model
    model, tokenizer = load_model()
    
    # User input
    user_input = st.text_input("Ask your question here:")
    
    if user_input:
        # Capitalize first letter
        formatted_input = user_input[0].upper() + user_input[1:]
        
        # Generate response
        response = generate_response(formatted_input, model, tokenizer)
        
        # Display response
        st.subheader("Answer:")
        st.write(response)

if __name__ == "__main__":
    main()
