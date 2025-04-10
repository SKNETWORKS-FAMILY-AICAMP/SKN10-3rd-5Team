
import streamlit as st
from streamlit_markdown import st_streaming_markdown


def add_history(message_history, role, content):
  message_history.append({"role": role, "content": content})
  
  return message_history


def write_history(messages):
  for message in messages[1:]:  # system 메시지 제외
    role = message["role"]
    if role == "user":
      write_chat(role, message["content"], is_stream=False)
    else:
      write_chat(role, message["content"], is_stream=True)


def write_chat(role, message, is_stream=False):
  with st.chat_message(role):
    if is_stream:
      messages = st_streaming_markdown(message, key="token_stream")
    else:
      st.markdown(message)
      messages = message

  return messages
