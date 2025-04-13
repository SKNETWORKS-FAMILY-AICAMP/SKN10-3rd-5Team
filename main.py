from common.init import init
import streamlit as st

def main():
    init()

    st.set_page_config(page_title="🍳 재료 기반 요리 추천 챗봇", layout="wide")

    st.title("🍳 재료 기반 요리 레시피 추천 챗봇")
    st.caption("당신의 냉장고 속 재료로 현실 가능한 요리를 추천해주는 AI")

    # -------------------
    # 서비스 개요
    # -------------------
    st.header("1. 서비스 개요")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2920/2920244.png", width=300)

    with col2:
        st.subheader("\U0001F4AC 맞춤형 AI 요리 추천 챗봇")
        st.markdown("""
        이 챗봇은 사용자가 보유한 **재료**, **사용 가능한 조리도구**, **희망 조리 시간**을 입력하면, 
        실제 조리가 가능한 요리 레시피를 추천해주는 **AI 기반 맞춤형 요리 추천 서비스**입니다.

        ✅ **요리 이름을 몰라도 검색 가능**  
        ✅ **재료 대체, 시간·도구 제약 반영**  
        ✅ **LLM + 검색 기반(RAG) 구조로 현실적인 요리 추천**
        
        본 챗봇은 단순 추천을 넘어서 **실행 가능한 요리**를 중심으로 설계되었으며, 
        사용자의 조리 조건을 바탕으로 요리 난이도까지 고려한 결과를 제공합니다.
        """)

    st.divider()

    # -------------------
    # 기획 배경
    # -------------------
    st.header("2. 서비스 기획 배경")
    st.subheader("\U0001F9E0 사용자의 요리 문제를 해결합니다")

    st.markdown("""
    요리 레시피 검색은 여전히 키워드 중심의 방식에 머물러 있어, 
    사용자가 어떤 요리를 만들 수 있는지보다, **무엇을 만들고 싶은지**를 알고 있어야 접근할 수 있습니다.

    본 챗봇은 다음과 같은 문제를 해결하고자 기획되었습니다:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### ❌ 요리 이름을 모르면 검색 불가")
        st.caption("요리 키워드를 모르거나 너무 많은 정보 속에서 길을 잃기 쉬움")

    with col2:
        st.markdown("#### 🧊 재료 낭비")
        st.caption("냉장고 속 재료를 활용하지 못해 요리를 포기하는 상황 다수")

    with col3:
        st.markdown("#### ⏱️ 시간/도구 제한")
        st.caption("전자레인지나 후라이팬 등 제한된 환경에서 요리 추천이 어려움")

    st.success("이 서비스는 위와 같은 문제를 고려해, 실제 조리 가능한 요리만을 제안합니다.")

    st.divider()

    # -------------------
    # UX 시나리오
    # -------------------
    st.header("3. 사용자 경험 흐름 (UX 시나리오)")
    st.subheader("✅ 사용자는 조건만 입력하면 됩니다")

    user_col, bot_col = st.columns([1, 1])

    with user_col:
        st.markdown("""
        #### 👤 사용자 입력 예시
        - 보유 재료: 감자, 양파, 계란
        - 사용 도구: 전자레인지, 후라이팬
        - 희망 조리 시간: 10분 이내

        사용자는 키워드를 모를 때도 **재료 기반으로 요리를 찾을 수 있습니다**.
        """)

    with bot_col:
        st.markdown("""
        #### 🤖 챗봇 응답 예시
        🥔🍳 감자달걀전 요리를 추천드립니다!

        1. 감자를 강판에 갈아 수분 제거
        2. 계란과 반죽하여 팬에 부치기
        3. 앞뒤로 노릇하게 익히기

        ✅ 부침가루 없이도 가능 / ⏱️ 10분 내 조리 가능
        ✅ **대체 재료** 안내까지 포함됨
        """)

    st.divider()

    # -------------------
    # 기술 구성 요소
    # -------------------
    st.header("4. 기술 구성 요소")
    st.subheader("\U0001F9D0 어떻게 동작하나요?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🧠 LLM 기반 응답 생성
        - Fine-tuning된 **Gemma 3 / LLaMA 3** 모델 사용
        - 실제 자막 기반 조리 흐름 학습 (YouTube 자막 기반)

        #### 🔍 RAG (검색 기반 QA)
        - 재료·도구·시간을 바탕으로 벡터 DB 검색
        - 검색된 자막으로 LLM이 최종 응답 생성
        """)

    with col2:
        st.markdown("""
        #### ⚙️ 응답 평가 및 선택
        - **응답 A**: LLM이 직접 생성
        - **응답 B**: 검색(RAG) 기반 생성
        - 두 결과를 GPT-4o-mini로 평가 → 더 나은 쪽 제공

        #### 📦 Embedding + Vector DB
        - OpenAI 임베딩 모델 사용 (소형)
        - FAISS를 활용한 벡터 기반 유사도 검색
        """)

    st.divider()

    # -------------------
    # 데이터 소스
    # -------------------
    st.header("5. 데이터 소스")
    st.subheader("\U0001F4FD️ 유튜브 자막 기반 현실 레시피")

    st.markdown("""
    신뢰도 높은 다음 유튜브 채널 자막을 기반으로 Fine-tuning 및 검색 데이터셋을 구축했습니다:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.write("- [딸을 위한 레시피](https://www.youtube.com/@%EB%94%B8%EC%9D%84%EC%9C%84%ED%95%9C%EB%A0%88%EC%8B%9C%ED%94%BC)")
        st.write("- [백종원의 요리비책](https://www.youtube.com/playlist?list=PLoABXt5mipg4vxLw0NsRQLDDVBpOkshzF)")

    with col2:
        st.write("- [백종원의 쿠킹로그](https://www.youtube.com/playlist?list=PLoABXt5mipg6mIdGKBuJlv5tmQFAQ3OYr)")
        st.write("- [한 가지 재료로 N가지 요리](https://www.youtube.com/playlist?list=PL7T0UWXKNl7TXa5t6I2qJ6vWtJhTSpRyt)")

    st.success("\U0001F916 당신의 냉장고 속 재료로, AI가 오늘의 저녁을 추천해드립니다!")

if __name__ == "__main__":
    main()
