import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import requests
import zipfile
import io
import os

# --- Configuration ---
MODEL_DIR_NAME = "DistilGPT2_Model_local"  # Local directory to store the downloaded model
GITHUB_REPO_URL = "https://github.com/MarpakaPradeepSai/Advanced-Events-Ticketing-Customer-Support-Chatbot"
GITHUB_MODEL_DIR_PATH = "DistilGPT2_Model"  # Path to your model directory in the repo
MODEL_ZIP_NAME = "model_files.zip"  # Name of the zip file we'll create and download

# --- Helper Functions ---

@st.cache_resource  # Use cache_resource for model loading (once per app run)
def load_model_and_tokenizer(model_dir):
    """Loads the model and tokenizer from the specified directory."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_length=256):
    """Generates a chatbot response."""
    device = model.device  # Get the model's device (CUDA or CPU)
    model.eval()

    input_text = f"Instruction: {instruction} Response:"

    # Tokenize the input and move the tensors to the same device as the model
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

    # Extract the response part after "Response:"
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

def download_and_extract_model(repo_url, model_dir_path, local_dir, zip_name):
    """Downloads the model files as a zip from GitHub and extracts them."""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    zip_url = f"{repo_url}/archive/main.zip"  # URL to download the main branch as zip
    try:
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract the specific model directory from the zip
            model_zip_path = f"{repo_name_from_url(repo_url)}-main/{model_dir_path}/"
            for zip_info in z.infolist():
                if zip_info.filename.startswith(model_zip_path):
                    # Extract to the local directory, removing the repo and model path prefix
                    target_path = zip_info.filename[len(model_zip_path):]
                    if target_path: # avoid extracting directory entries
                        z.extract(zip_info, path=local_dir)
                        extracted_path = os.path.join(local_dir, target_path)
                        print(f"Extracted: {extracted_path}") # Optional: Print extracted files


        print(f"Model files downloaded and extracted to '{local_dir}'")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model files from GitHub: {e}")
        return False
    except zipfile.BadZipFile as e:
        st.error(f"Error extracting zip file: {e}. Corrupted zip file?")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during download/extraction: {e}")
        return False

def repo_name_from_url(repo_url):
    """Extracts the repository name from a GitHub URL."""
    return repo_url.split("/")[-1]


# --- Streamlit App ---

st.title("Advanced Ticketing Chatbot")

# 1. Download and Extract Model (only if not already present)
if not os.path.exists(MODEL_DIR_NAME):
    with st.spinner("Downloading and loading model from GitHub..."):
        repo_name = repo_name_from_url(GITHUB_REPO_URL)
        downloaded = download_and_extract_model(GITHUB_REPO_URL, GITHUB_MODEL_DIR_PATH, MODEL_DIR_NAME, MODEL_ZIP_NAME)
        if not downloaded:
            st.stop() # Stop if download fails

# 2. Load Model and Tokenizer
try:
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR_NAME)
except Exception as e:
    st.error(f"Error loading model and tokenizer: {e}")
    st.stop()


# 3. Chat Interface
user_input = st.text_area("Enter your question:", placeholder="How to cancel my ticket?", height=100)

if st.button("Get Response"):
    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Capitalize the first letter of the instruction
                instruction = user_input[0].upper() + user_input[1:]
                response = generate_response(model, tokenizer, instruction)
                st.write("### Chatbot Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a question.")


# --- Instructions for Deployment (Shown in the Streamlit App) ---
st.markdown("---")
st.markdown("## Instructions for Streamlit Deployment:")
st.markdown("""
1.  **GitHub Repository:** Make sure this `streamlit_app.py` file and your `DistilGPT2_Model` directory (containing model files) are in your GitHub repository.
2.  **Streamlit Cloud:**
    *   Go to [Streamlit Cloud](https://share.streamlit.io/).
    *   Connect to your GitHub repository.
    *   Select your repository and branch.
    *   Set the **App file** to `streamlit_app.py`.
    *   **Advanced settings:**
        *   **Python version:** Choose the Python version you used for development (ideally 3.8 or higher).
        *   **Dependencies:** Streamlit Cloud will automatically detect `requirements.txt` if you have one. If not, it will try to infer dependencies.  It's best to create a `requirements.txt` file in your repository root with the following:

            ```
            streamlit
            transformers
            torch
            requests
            ```
    *   Click "Deploy!".

3.  **Wait:** Streamlit Cloud will build and deploy your app. This might take a few minutes the first time as it downloads the model files.

**Important Notes:**

*   **Model Size:**  DistilGPT2 is relatively small, but if your fine-tuned model is very large, deployment on free Streamlit Cloud might be slow or hit memory limits. Consider optimizing your model or using a different hosting solution for larger models.
*   **GitHub Access:** Streamlit Cloud needs access to your public GitHub repository to deploy. If your repository is private, you'll need to configure Streamlit Cloud accordingly.
*   **Error Handling:** The code includes basic error handling, but you might want to add more robust error logging and user feedback for a production application.
""")
