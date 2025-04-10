
import streamlit as st
from streamlit_markdown import st_streaming_markdown

def write_history(messages):
    for message in messages[1:]:  # system 메시지 제외
        role = message["role"]
        if role == "user":
            write_chat(role, message["content"])
        else:
            write_chat(role, message["content"])

def write_chat(role, message):
    with st.chat_message(role):
        if not isinstance(message, str):
            message_placeholder = st.empty()
            messages = ""
            for msg in message:
                messages += msg                   
                message_placeholder.markdown(messages)
        else:
            st.markdown(message)
            messages = message

    return messages