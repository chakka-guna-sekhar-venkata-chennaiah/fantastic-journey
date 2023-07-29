import os
import streamlit as st
import requests
import json
from PIL import Image
import base64
from io import BytesIO


# Helper function to call API endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=st.secrets['HF_API_TTI_BASE']):
    hf_api_key = st.secrets['HF_API_KEY']
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }   
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers=headers,
                                data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))

# Helper function for image processing
def image_to_base64_str(pil_image):
    byte_arr = BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

# Streamlit app code
def main():
    st.markdown("# Image Generation with Stable Diffusion")
    prompt = st.text_input("Enter your prompt here...")
    negative_prompt = st.text_input("Enter negative prompt here...")
    steps = st.slider("Inference Steps", min_value=1, max_value=100, value=25, step=1)
    guidance = st.slider("Guidance Scale", min_value=1, max_value=20, value=7, step=1)
    width = st.slider("Width", min_value=64, max_value=512, value=512, step=64)
    height = st.slider("Height", min_value=64, max_value=512, value=512, step=64)

    if st.button("Submit"):
        result_image = generate(prompt, negative_prompt, steps, guidance, width, height)
        st.image(result_image, caption="Generated Image", use_column_width=True)

        # Download button for the generated image
        download_button_str = download_button(result_image, "Download Generated Image", "generated_image.png")
        st.markdown(download_button_str, unsafe_allow_html=True)

# Other functions
def generate(prompt, negative_prompt, steps, guidance, width, height):
    inputs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height
    }
    result = get_completion(inputs)
    generated_image = base64_to_pil(result['image'])
    return generated_image

# Function to generate download button link
def download_button(image, download_text, filename):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{download_text}</a>'
    return href

if __name__ == "__main__":
    main()
