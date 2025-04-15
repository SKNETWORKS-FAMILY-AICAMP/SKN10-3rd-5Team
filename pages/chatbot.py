import streamlit as st
import time
from common.init import get_question, init
from common.display import write_history
from common.ask import ask

def app():
    st.set_page_config(layout="centered")
    init()
    st.title("Chatbot")

    write_history(st.session_state.messages)

    cooking_time, cooking_tools, question = get_question()
    if (cooking_time is None) or (len(cooking_tools) == 0):
        st.warning("옵션을 선택해주세요.")

    if question:
        # 진행률 표시 컴포넌트
        progress_bar = st.progress(0)
        percent_text = st.empty()
        time_text = st.empty()
        start_time = time.time()

        # 진행률 콜백 함수
        def progress_callback(current, total):
            percent = int((current / total) * 100)
            elapsed = time.time() - start_time
            estimated = elapsed * (total / current) if current > 0 else 0
            
            progress_bar.progress(percent)
            percent_text.markdown(f"**진행률: {percent}%**")
            time_text.markdown(
                f"⏱️ 경과: {elapsed:.1f}초 | 예상: {estimated:.1f}초"
            )

        # 답변 생성
        with st.spinner("요리 레시피를 생성 중입니다..."):
            st.session_state.messages = ask(
                question=question,
                message_history=st.session_state.messages,
                cooking_time=cooking_time,
                cooking_tools=cooking_tools,
                progress_callback=progress_callback
            )

        # 컴포넌트 정리
        progress_bar.empty()
        percent_text.empty()
        time_text.empty()

if __name__ == "__main__":
    app()