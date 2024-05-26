from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.llms import Ollama
from langchain_groq import ChatGroq

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

from anthropic import Anthropic

import streamlit as st

import os
import io
import time
import base64
import cv2
import numpy as np

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]

# Upload file with profiles locally
## Will be updated to fetch from google drive directly later on
file_path = 'C:\\Users\\Bonoc\\Desktop\\EconLLM_TribesBot\\Search + RAG - Profiles (1).txt'

@st.cache_data
def parse_tribal_profiles(file_path):
    """
    Parses a text file containing tribal profiles separated by '#' symbols.

    :param file_path: Path to the text file containing the profiles.
    :return: A list of tuples, each containing the name and profile of a tribe.
    """
    # Open and read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content based on the delimiter "##############"
    tribes_raw = content.split('##############')

    # Process each segment to extract name and profile
    tribes = []
    for tribe_raw in tribes_raw:
        if tribe_raw.strip():  # Ignore empty segments
            # Split by newline and filter out empty lines
            lines = list(filter(bool, map(str.strip, tribe_raw.split('\n'))))
            if lines:  # If there are lines present after filtering
                tribe_name = lines[0]
                tribe_profile = ' '.join(lines[1:])  # Join the remaining lines to form the profile
                tribes.append((tribe_name, tribe_profile))

    return tribes

@st.cache_data
def tribes_to_dict():
    """
    Converts a list of tribal profiles to a dictionary.

    :param tribes: A list of tuples, each containing the name and profile of a tribe.
    :return: A dictionary where the keys are the tribe names and the values are the profiles.
    """
    # this variable now has all the profiles in the text file (works for all three methods because they share the same delimiter)
    tribal_profiles = parse_tribal_profiles(file_path)

    data = {'tribe_names': [],
            'profiles': []}

    for name, profile in tribal_profiles:
        data['tribe_names'].append(name)
        data['profiles'].append(profile)
        
    return data


def llm_response_generator(llm_response):
    for word in llm_response.split():
        yield word + (' ' if word != '\n' else '\n')
        time.sleep(0.05)

@st.cache_resource
def load_llm(provider="llama3-70b-8192",
             temperature=0.55,
             max_tokens=150):
    """
    Loads a language model based on the specified provider.

    Parameters:
    - provider (str, optional): The LLM provider and version. Default is 'openai-3.5'.
    - temperature (float, optional): The generation temperature for the LLM. Default is 1.
    - max_tokens (int, optional): The maximum token limit for LLM responses. Default is 40.
    - profile (str, optional): The cultural profile to build system prompt. Default is no system prompt (vainilla LLM behavior)

    Returns:
    - ChatOpenAI: An instance of the LangChain object ChatOpenAI.

    Raises:
    - ValueError: If an unsupported provider is specified.
    """
    supported_providers = ['gpt-3.5-turbo', 'gpt-4-turbo', 'llama3-70b-8192']

    if provider not in supported_providers:
        raise ValueError(f"Unsupported provider. Supported providers are: {', '.join(supported_providers)}")

    if 'gpt' in provider:
        llm = ChatOpenAI(model=provider,
                     temperature=temperature,
                     max_tokens=max_tokens)

    else: 
        llm = ChatGroq(model="llama3-70b-8192",
                       temperature=temperature,
                       max_tokens=max_tokens)

    return llm

@st.cache_resource
def create_chain(_llm, SYS_PROMPT):
    
    output_parser = StrOutputParser()
                
    prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=SYS_PROMPT
                ),
                
                HumanMessagePromptTemplate.from_template(
                    "{chat_history}"
                ),  # Where the human input will injected
            ]
        )
            
    chat_llm_chain = LLMChain(llm=_llm,
                              prompt=prompt,
                              output_parser=output_parser)
    
    return chat_llm_chain

def get_base64_encoded_image(image_bytes):
    base_64_encoded_data = base64.b64encode(image_bytes)
    base64_string = base_64_encoded_data.decode('utf-8')
    return base64_string

def process_image_with_anthropic(image_bytes):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    MODEL_NAME = "claude-3-opus-20240229"
    
    base64_image = get_base64_encoded_image(image_bytes)
    message_list = [
        {
            "role": 'user',
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                {"type": "text", "text": "Describe the content of this image clearly."}
            ]
        }
    ]

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=message_list
    )
    return response.content[0].text

def process_image_with_ollama(image_bytes):
    ollama = Ollama(model="moondream")
    base64_image = get_base64_encoded_image(image_bytes)
    llm_with_image_context = ollama.bind(images=[base64_image])
    
    response = llm_with_image_context.invoke("Describe the content of this image in detail. Provide a structure breakdown of the image.")
    return response

@st.cache_resource
def load_hf_moondream():
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda")
    else:
        print("GPU is not available")
        device = torch.device("cpu")

    # Load the model and move the model to the GPU if available
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    
    return model, tokenizer, device

def process_image_with_hf_moondream(image_bytes):
    
    
    model, tokenizer, device = load_hf_moondream()
    
    image = Image.open(io.BytesIO(image_bytes))
    enc_image = model.encode_image(image).to(device)
    
    response = model.answer_question(enc_image, "Describe this image briefly.", tokenizer)
    return response

@st.experimental_fragment
def get_user_image():
    st.markdown("### **Share a picture:**")
    img_file_buffer = st.camera_input("Take the picture",
                                    key="User image input")
    
    # add option to upload image

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 
                            cv2.IMREAD_COLOR)
        
        st.write("### **Captured Image**")
        st.image(cv2_img, channels="BGR")
        st.write(cv2_img.shape)
            
        if st.session_state.last_image_processed != bytes_data:
        
            with st.spinner('Processing image with Claude 3...'):
                #image_description = process_image_with_hf_moondream(bytes_data)
                #image_description = process_image_with_ollama(bytes_data)
                
                image_description = process_image_with_anthropic(bytes_data)
                st.session_state.image_description = image_description
                
                st.session_state.messages.append({"role": "Agent", "content": f"Image description: {image_description}"})
                with st.chat_message("Agent"):
                    st.write(f"Image description: {image_description}")
                    
                st.session_state.last_image_processed = bytes_data
        
        img_file_buffer = None