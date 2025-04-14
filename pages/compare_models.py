import streamlit as st

from common.init import init
from common.display import write_history
from common.ask import ask
from common.rag import create_rag_chain
from llm.ollama import Provider_Ollama

def app():
    st.set_page_config(layout="wide")
    init(["messages_1", "messages_2"])
    st.title("Compare Models")
    st.caption("**예시 질문**")
    st.caption("양파, 마늘, 그리고 계란을 사용해서 20분 이내에 후라이팬만 이용해 만들 수 있는 요리를 추천해주세요.")

    col1, col2 = st.columns(2)

    question = st.chat_input("무엇이든지 물어봐주세요.")

    with col1:
        st.subheader("Gemma3 4b Model")
        write_history(st.session_state.messages_1)

        if question:
            # 첫 번째 모델 설정
            provider = Provider_Ollama()
            model_1 = provider("gemma3_4b_q8")
            st.session_state.messages_1 = ask(
                question=question, 
                message_history=st.session_state.messages_1,
                llm_model=model_1
            )
            # Store messages after ask function
            msg_1 = st.session_state.messages_1.copy()
            st.write(msg_1)
            # Reset session state
            st.session_state.messages_1 = []
        
    with col2:
        st.subheader("Fine-Tuned Model")
        write_history(st.session_state.messages_2)
    
        if question:
            # 두 번째 모델 설정
            provider = Provider_Ollama()
            model_2 = provider("gemma3_4b_q8_recipe")
            st.session_state.messages_2 = ask(
                question=question, 
                message_history=st.session_state.messages_2,
                llm_model=model_2
            )
            # Store messages after ask function
            msg_2 = st.session_state.messages_2.copy()
            st.write(msg_2)
            # Reset session state
            st.session_state.messages_2 = []

if __name__ == "__main__":
    app()
