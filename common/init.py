import streamlit as st
from dotenv import load_dotenv

def init():
    load_dotenv()

    if "messages" not in st.session_state:
        st.session_state.messages = []

def get_question():
    # 사이드바에 선택박스 추가
    with st.sidebar:
        # 왼쪽 선택박스: 요리 시간 선택 (기본값을 None으로 설정)
        cooking_time = st.selectbox(
            "요리 시간을 선택하세요:", 
            [None, "10분 이하", "20분 이하", "30분 이하"]
        )

        # 오른쪽 선택박스: 조리 도구 선택 (기본값을 None으로 설정)
        cooking_tools = st.multiselect(
            "사용 가능한 조리 도구를 모두 선택하세요:",
            ['프라이팬', '냄비', '에어프라이어', '주걱', '도마', '칼', '믹서기', '채반', '가스레인지', '오븐'],
            default=None
        )
    
    # 선택된 값 출력 (디버깅용)
    print("=" * 50)
    print("요리 시간:", cooking_time)
    print("조리 도구:", cooking_tools)
    print("=" * 50)
    
    # 질문 입력 받기
    question = st.chat_input("무엇이든지 물어봐주세요.")
    print("=" * 50)
    print(question)
    print("=" * 50)
    
    return cooking_time, cooking_tools, question
