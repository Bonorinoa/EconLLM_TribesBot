from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

import streamlit as st

import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

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

# Let's try gpt-3.5-turbo, gpt-4, and two open-source models from hugging face in the future.
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def load_llm(provider='gpt-3.5-turbo',
                temperature=1,
                max_tokens=40):
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
    supported_providers = ['gpt-3.5-turbo', 'gpt-4']

    if provider not in supported_providers:
        raise ValueError(f"Unsupported provider. Supported providers are: {', '.join(supported_providers)}")

    llm = ChatOpenAI(model=provider,
                     temperature=temperature,
                     max_tokens=max_tokens)

    return llm

def create_chain(llm, SYS_PROMPT):
                
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
            

    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )
    
    return chat_llm_chain