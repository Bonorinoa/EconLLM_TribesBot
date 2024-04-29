import streamlit as st
from utils import parse_tribal_profiles, load_llm, create_chain

import time

# Upload file with profiles locally
## Will be updated to fetch from google drive directly later on
file_path = 'C:\\Users\\Bonoc\\Desktop\\EconLLM_TribesBot\\Search + RAG - Profiles (1).txt' # in drive under profiles folder

# this variable now has all the profiles in the text file (works for all three methods because they share the same delimiter)
tribal_profiles = parse_tribal_profiles(file_path)

data = {'tribe_names': [],
        'profiles': []}

for name, profile in tribal_profiles:
    print(f"Tribe Name: {name}\nProfile: {profile}\n\n")
    data['tribe_names'].append(name)
    data['profiles'].append(profile)
    
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi!"}]

def llm_response_generator(llm_response):
    for word in llm_response.split():
        yield word + (' ' if word != '\n' else '\n')
        time.sleep(0.05)
    
with st.sidebar:
    st.write("Tribes")
    tribe_selected = st.selectbox("Select a tribe", data['tribe_names'])
    
    st.button('Clear Chat History', on_click=clear_chat_history)
    
# profile for the selected tribe. Find index of selected tribe and display the profile
#st.write("Profile")
target_profile = data['profiles'][data['tribe_names'].index(tribe_selected)]

system_prompt = f"""You are a member of the {tribe_selected} tribe with the following characteristics:

                -----
                {target_profile}

                -----

    """

llm = load_llm(provider='gpt-4', 
               temperature=0.85, 
               max_tokens=250)

if 'messages' not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            

chat_llm_chain = create_chain(llm, system_prompt)

if user_input := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(user_input)
            
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
                    
            chat_response = chat_llm_chain.predict(chat_history=st.session_state.messages)
            
            st.write_stream(llm_response_generator(chat_response))
    
    st.session_state.messages.append({"role": "assistant", "content": chat_response})