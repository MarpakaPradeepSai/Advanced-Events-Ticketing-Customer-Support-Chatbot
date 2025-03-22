import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Model and tokenizer loading function
@st.cache(allow_output_mutation=True)
def load_model():
    model_dir = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot/raw/main/DistilGPT2_Model"
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return tokenizer, model

# Response generation function
def generate_response(instruction, max_length=256):
    tokenizer, model = load_model()
    device = model.device
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

# Streamlit app interface
st.title("Advanced Events Ticketing Chatbot")
st.write("Ask the chatbot about ticketing queries!")

user_input = st.text_input("Your Query:", "")

if st.button("Get Response"):
    if user_input.strip():
        user_input = user_input[0].upper() + user_input[1:]  # Capitalize first letter
        with st.spinner("Generating response..."):
            response = generate_response(user_input)
        st.success(f"Chatbot: {response}")
    else:
        st.error("Please enter a valid query.")
