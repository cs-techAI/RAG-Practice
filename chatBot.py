import streamlit as st
import openai

st.title("Clone Bot")

openai.api_key = ""

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):    # used walrus operator to assign the user input to the prompt variable
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})


    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = openai.ChatCompletion.create(
            model = st.session_state["openai_model"],
            messages = [
                {"role": m["role"], "content" :m["content"]}
                for m in st.session_state.messages
                ],
        )
        full_response += response.choices[0].message.content      
        message_placeholder.markdown(full_response + "🎹")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})