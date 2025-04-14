import streamlit as st

from common.init import init
from common.display import write_history
from common.ask import ask

def app():
    init(["messages_1", "messages_2"])
    st.title("Compare Models")
    st.caption("**예시 질문**")
    st.caption("양파, 마늘, 그리고 계란을 사용해서 20분 이내에 후라이팬만 이용해 만들 수 있는 요리를 추천해주세요.")

    col1, col2 = st.columns(2)

    question = st.chat_input("무엇이든지 물어봐주세요.")

    with col1:
        st.subheader("Gemma3 4b Model")

        write_history(st.session_state.messages_1)
        # question1 = st.chat_input("무엇이든지 물어봐주세요.", key="input_1")

        if question:
            st.session_state.messages_1 = ask(
                question=question, 
                message_history=st.session_state.messages_1
            )

    with col2:
        st.subheader("Fine-Tuned Model")

        write_history(st.session_state.messages_2)
        # question2 = st.chat_input("무엇이든지 물어봐주세요.", key="input_2")
    
        if question:
            st.session_state.messages_2 = ask(
                question=question, 
                message_history=st.session_state.messages_2
            )

if __name__ == "__main__":
    app()
