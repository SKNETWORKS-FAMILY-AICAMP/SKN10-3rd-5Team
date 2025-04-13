
import streamlit as st

from common.init import get_question, init
from common.display import write_history
from common.ask import ask


def app():
    init()
    st.title("Chatbot")

    write_history(st.session_state.messages)

    cooking_time, cooking_tools, question = get_question()
    if (cooking_time is None) or (len(cooking_tools) == 0):
            st.warning("옵션을 선택해주세요.")
            
    if question:
        st.session_state.messages = ask(question=question, message_history=st.session_state.messages)


if __name__ == "__main__":
    app()
