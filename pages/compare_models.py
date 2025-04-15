import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage

from common.init import init
from common.display import write_history

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
        
        for message in st.session_state.messages_1:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if question:
            with st.chat_message("user"):
                st.write(question)

            model_1 = ChatOllama(
                model="gemma3-q8",
                model_kwargs={"max_tokens": 1000},
                streaming=True
            )

            messages = [HumanMessage(content=question)]

            with st.chat_message("assistant"):
                response_stream = model_1.stream(messages)
                full_response = ""
                response_container = st.empty()

                for chunk in response_stream:
                    if isinstance(chunk, AIMessage):
                        full_response += chunk.content
                        response_container.markdown(full_response)

                response_container.markdown(full_response)

            st.session_state.messages_1.append({"role": "user", "content": question})
            st.session_state.messages_1.append({"role": "assistant", "content": full_response})

    with col2:
        st.subheader("Fine-Tuned Model")
        
        for message in st.session_state.messages_2:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if question:
            with st.chat_message("user"):
                st.write(question)

            model_2 = ChatOllama(
                model="gemma3-recipe",
                model_kwargs={"max_tokens": 1000},
                streaming=True
            )

            messages = [HumanMessage(content=question)]

            with st.chat_message("assistant"):
                response_stream = model_2.stream(messages)
                full_response = ""
                response_container = st.empty()

                for chunk in response_stream:
                    if isinstance(chunk, AIMessage):
                        full_response += chunk.content
                        response_container.markdown(full_response)

                response_container.markdown(full_response)

            st.session_state.messages_2.append({"role": "user", "content": question})
            st.session_state.messages_2.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    app()
