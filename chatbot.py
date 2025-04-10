
import streamlit as st

from common.init import get_question, init
from common.history import write_history
from common.ask import ask


def app():
  init()

  write_history(st.session_state.messages)

  question = get_question()

  if question:
    st.session_state.messages = ask(question=question, message_history=st.session_state.messages)


if __name__ == "__main__":
  app()
