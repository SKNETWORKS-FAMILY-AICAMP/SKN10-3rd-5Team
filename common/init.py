
import streamlit as st
from dotenv import load_dotenv


def init():
  load_dotenv()

  if "messages" not in st.session_state:
    st.session_state.messages = []

  st.title("Chatbot")


def get_question():
  question = st.chat_input("무엇이든지 물어봐주세요.")
  return question
