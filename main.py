import streamlit as st
from utils import (load_llm, create_chain, get_user_image, 
                   tribes_to_dict, llm_response_generator,
                   process_image_with_hf_moondream, parse_json_profiles)

import time


# TODO: Add dropdown of game prompts

#data = tribes_to_dict()
data = parse_json_profiles()
 
    
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi!"}]

def main():
    st.title("Multimodal Endowment Effect")
    
    if "last_image_processed" not in st.session_state:
        st.session_state.last_image_processed = None
        
    if "image_description" not in st.session_state:
        st.session_state.image_description = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
        
    with st.sidebar:
        st.markdown("### Select a tribe")
        tribe_selected = st.selectbox("Select a tribe", data['tribe_names'] + ["GPT-4"])
        
        st.markdown("### LLM parameters")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.65)
        max_tokens = st.slider("Max tokens", 50, 300, 150)
        
        get_user_image()
        
        st.button('Clear Chat History', on_click=clear_chat_history)
        
    # profile for the selected tribe. Find index of selected tribe and display the profile
    #st.write("Profile")
    llm = load_llm(provider='llama3-70b-8192', 
                   temperature=temperature, 
                   max_tokens=max_tokens)
    
    print(llm.temperature, llm.max_tokens)

    if tribe_selected != "GPT-4":
        target_profile = data['claude-3-opus-20240229'][tribe_selected]

        system_prompt = f"""You are a member of the {tribe_selected} tribe with the following characteristics:

                        -----
                        {target_profile}

                        -----
                        
                        You have been randomly selected to participate in an economic experiment.
                        Do not deviate from the topic of the conversation intiated by the experimenter.
                        Provide honest and truthful responses to the questions asked based on the information about your tribe.

            """
            
    else:
        system_prompt = ("You are a helpful assistant that has been randomly selected to participate in an economic experiment."
                         "Do not deviate from the topic of the conversation intiated by the experimenter.")
                

    chat_llm_chain = create_chain(llm, 
                                  system_prompt)

    if user_input := st.chat_input("Type a message..."):
        st.session_state.messages.append({"role": "user", 
                                          "content": user_input})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(user_input)
                
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                        
                chat_response = chat_llm_chain.predict(chat_history=st.session_state.messages)
                
                st.write_stream(llm_response_generator(chat_response))
        
            st.session_state.messages.append({"role": "assistant", "content": chat_response})
    
    #print("---- Chat history ----")
    #print(st.session_state.messages)
        
if __name__ == "__main__":
    main()