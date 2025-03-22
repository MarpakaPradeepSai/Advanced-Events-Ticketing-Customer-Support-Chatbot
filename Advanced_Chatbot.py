import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

@st.cache(allow_output_mutation=True)
def load_model():
    model_url = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"
    
    model = GPT2LMHeadModel.from_pretrained(model_url)
    tokenizer = GPT2Tokenizer.from_pretrained(model_url)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, device

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

# Streamlit UI
st.title("Advanced Events Ticketing Chatbot")
st.write("Ask me about ticketing, cancellations, and more!")

model, tokenizer, device = load_model()

user_input = st.text_input("Your Question:", "")

if user_input:
    user_input = user_input[0].upper() + user_input[1:]  # Capitalize first letter
    with st.spinner("Generating response..."):
        response = generate_response(model, tokenizer, device, user_input)
    st.subheader("Chatbot Response:")
    st.write(response)
