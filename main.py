from common.init import init
import streamlit as st

def main():
    init()

    st.set_page_config(page_title="🍳 요리왕 좌룡", layout="wide")

    st.title("요리왕 좌룡")
    st.caption("재료 기반 요리 레시피 추천 챗봇 서비스")

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
        이 서비스는 사용자가 가지고 있는 **재료**, **조리도구**, **희망 조리 시간**등의 조건을 입력하면, 이를 기반으로 실제 조리가 가능한 레시피를 추천해주는 **AI 기반 맞춤형 요리 추천 챗봇**입니다.

        ✅ **요리 이름을 몰라도 검색 가능**  
        ✅ **재료 대체, 시간·도구 제약 반영**  
        ✅ **LLM + 검색 기반(RAG) 구조로 현실적인 요리 추천**
        
        기존의 요리 이름 기반 검색 방식에서 벗어나, LLM과 RAG 기술을 활용하여 재료 중심의 요리 추천이라는 새로운 접근 방식을 제시합니다.
        """)

    st.divider()

    # -------------------
    # 기획 배경
    # -------------------
    st.header("2. 서비스 기획 배경")
    st.subheader("\U0001F9E0 사용자의 요리 문제를 해결합니다")

    st.markdown("""
    기존 레시피 검색은 여전히 키워드 중심의 방식에 머물러 있어, 
    사용자가 어떤 요리를 만들 수 있는지보다, **무엇을 만들고 싶은지**를 알고 있어야 접근할 수 있습니다.

    본 챗봇은 다음과 같은 문제를 해결하고자 기획되었습니다:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### ❌ 기존 레시피 검색의 한계")
        st.caption("사용자가 요리 이름이나 키워드를 모르면 검색이 어려움")

    with col2:
        st.markdown("#### 🧊 현실적인 사용자 불편")
        st.caption("냉장고 속 재료를 활용하지 못해 요리를 포기하는 상황 다수")

    with col3:
        st.markdown("#### ⏱️ 사용자 제약 조건 고려 부족")
        st.caption("자취생, 바쁜 직장인 등 도구나 시간 제약 반영 어려움")

    st.success("이 서비스는 위와 같은 문제를 고려해, 유튜브 요리 영상의 자막 데이터를 기반으로 LLM을 파인튜닝하거나 RAG 방식으로 검색하여, 현실적으로 조리 가능한 레시피를 추천하는 실행 가능한 AI 요리 서비스를 지향합니다.")

    st.divider()

    # -------------------
    # UX 시나리오
    # -------------------
    st.header("3. 사용자 경험 흐름 (UX 시나리오)")
    st.subheader("✅ 사용자는 조건만 입력하면 됩니다")

    user_col, bot_col = st.columns([1, 1])

    with user_col:
        st.markdown("""
        #### 👤 사용자 입력 단계
        - 보유 재료: 예) 감자, 양파, 계란 (텍스트 입력)
        - 사용 가능한 조리 도구: 예) 후라이팬, 전자레인지 (Select Box)
        - 희망 조리 시간: 예) 10분 이하, 30분 이하 (Select Box)

        """)
        st.markdown("➡️ ‘전송’ 버튼 클릭하면, 챗봇이 요리 추천을 시작합니다.")

    with bot_col:
        st.markdown("""
        #### 🤖 챗봇 응답 예시
        🥔🍳 감자달걀전 요리를 추천드립니다!

        부침가루가 없는 경우, 계란과 감자 전분으로 대체 가능합니다.

        [요리 순서]
        1. 감자를 강판에 갈아 수분 제거
        2. 계란을 풀어 감자와 반죽
        3. 팬에 기름을 두르고 약불에서 부치기
        4. 앞뒤로 노릇하게 익히기
        5. 간장 + 식초 찍어 먹기

        맛있게 드세요~ 😋
        """)

        st.markdown("➡️ 챗봇은 단순한 텍스트 레시피를 넘어서, 대체 가능한 재료 안내, 조리 순서 제공, 사용자 조건에 맞춘 최적화된 제안을 제공합니다.")
    st.divider()

    # -------------------
    # 기술 구성 요소
    # -------------------
    st.header("4. 기술 구성 요소")
    st.subheader("\U0001F9D0 어떻게 동작하나요?")

    st.markdown("""
    #### 🧠 LLM 모델
    - 파인튜닝 대상: gemma3-4b Model

    #### 🔍 검색 기반 QA (RAG 방식)
    - 사용자 입력값(재료, 조리도구, 시간)을 기반으로 관련 자막을 검색
    - 검색 결과를 LLM에 제공하여 맥락 이해 + 응답 생성
                
    #### 🧠  Embedding 모델
    - OpenAI의 소형 embedding 모델 (텍스트 벡터화 용도)

    #### 🔍 Vector DB
    - FAISS 활용, 유사한 요리 레시피 클러스터링 및 검색        
    """)

    st.divider()

    # -------------------
    # 데이터 소스
    # -------------------
    st.header("5. 데이터 소스")
    st.subheader("\U0001F4FD️ 유튜브 자막 기반 레시피")

    st.markdown("""
    📌신뢰도 높은 유튜브 채널 자막을 기반으로 Fine-tuning 및 검색 데이터셋을 구축했습니다:
    
    선택된 채널들은 자막 품질이 비교적 우수하거나 실제 요리 흐름이 잘 기록되어 있습니다.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.write("- [딸을 위한 레시피](https://www.youtube.com/@%EB%94%B8%EC%9D%84%EC%9C%84%ED%95%9C%EB%A0%88%EC%8B%9C%ED%94%BC)")
        st.write("- [백종원의 요리비책](https://www.youtube.com/playlist?list=PLoABXt5mipg4vxLw0NsRQLDDVBpOkshzF)")

    with col2:
        st.write("- [백종원의 쿠킹로그](https://www.youtube.com/playlist?list=PLoABXt5mipg6mIdGKBuJlv5tmQFAQ3OYr)")
        st.write("- [한 가지 재료로 N가지 요리](https://www.youtube.com/playlist?list=PL7T0UWXKNl7TXa5t6I2qJ6vWtJhTSpRyt)")

    st.success("\U0001F916 당신의 냉장고 속 재료로, AI가 오늘의 저녁을 추천해드립니다!")

    st.markdown("""
    ### RAG
    - [만개의 레시피](https://www.10000recipe.com/?srsltid=AfmBOoq_msLnwDaByNGMj6nSyX_i7IqxU3u43aEKLXwu5yqbxqN9foHs)

    📌 해당 사이트에서 필요한 정보를 `BeautifulSoup`을 통해 추출하여 사용하였습니다.
    """)

    st.image("https://github.com/user-attachments/assets/64ac3945-098f-495e-8a59-69d5f44d078c")

    st.divider()
    
    # -------------------
    # 데이터셋
    # -------------------
    st.header("5-1. 데이터셋 생성")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1) 데이터 수집")
        st.write("yt-dlp 라이브러리를 사용하여 유튜브 영상의 자막 데이터를 추출합니다.")

    with col2:
        st.subheader("2) 전처리 과정")
        st.write("- 자막 내 불필요한 요소 제거")
        st.write("- 요리와 직접적인 관련이 없는 단어 또는 문장 제거")
        st.write("- 음성 인식 오류로 잘못 표기된 단어 수정")
        st.write("- 특수문자 및 이모티콘 제거")
    
    with col3:
        st.subheader("3) 질문-답변 데이터셋 생성")
        st.write("- 전처리된 자막 데이터를 기반으로 gpt-4o-mini 모델을 사용하여 요리 과정을 중심으로 한 질문-답변 데이터셋을 생성한다.")     

    st.divider()

    # -------------------
    # 케이스별 활용
    # -------------------
    st.header("6. 사용자 케이스별 활용 예시")

    st.markdown("""
    🎯 케이스 유형         | 설명 |
    |-----------------------|------|
    | 🧊 냉장고 재료 기반  | 요리 이름 없이도 보유 재료만으로 추천 가능 |
    | 🍳 조리도구 기반     | 보유한 도구(전자레인지, 후라이팬 등)를 기반으로 한 요리 추천 |
    | ⏱️ 시간 부족         | 조리 시간 기반 필터링으로 바쁜 직장인도 활용 가능 |
    | 🧑‍🍳 요리 초보         | 상세한 조리 순서 제공으로 따라 하기 쉬움 |
    """)

    st.divider()

    # -------------------
    # 확장 및 아이디어
    # -------------------
    st.header("7. 확장 가능성 및 고도화 아이디어")

    st.markdown("""   
    | 기능                   | 설명 |
    |------------------------|------|
    | 🔁 연속 대화 지원     | “다른 요리는?”, “이 재료 빼고는?” 등의 후속 질의 처리 |
    | 📷 재료 이미지 인식   | 사진 업로드 시 재료 자동 추출 (OCR + 모델 결합) |
    | 🛒 장보기 연동        | 부족한 재료를 쇼핑몰 장바구니로 자동 연결 |
    | 💾 사용자 맞춤 저장   | 자주 쓰는 조리 조건, 재료, 도구 정보 자동 저장 후 추천 강화 |
    """)

    st.divider()

    # -------------------
    #  요약 및 기대 효과
    # -------------------
    st.header("8. 요약 및 기대 효과")
    st.markdown("이 챗봇은 현실적인 조리 제약을 고려한 개인 맞춤형 요리 추천이라는 명확한 사용자 가치를 제공합니다.")
    st.markdown("""- 입력은 간단하게, 출력은 풍부하고 실행 가능하게""")
    st.markdown("""- 재료 낭비를 줄이고, 요리 진입장벽을 낮추는 실용적인 AI 서비스""")

    st.divider()

    # -------------------
    #  응답 생성 및 서비스 동작 프로세스
    # -------------------
    st.header("9. 응답 생성 및 서비스 동작 프로세스")
    st.image('imgs/service_flow.png', width=600)

    st.subheader("""🔍 단계별 설명""")
    st.markdown("""
① 사용자 입력
                
    - 사용자가 조리 시간, 조리 도구 등의 정보를 포함하여 질문을 입력합니다.

                
② 요리 관련 여부 판단

    - 입력된 질문이 요리와 관련된 경우 ➡️ 레시피 추천 LLM이 답변 생성합니다.

    - 입력된 질문이 요리와 관련이 없는 경우 ➡️ 일반 LLM이 답변 생성합니다.


③ 레시피 추천 LLM 답변 적절성 평가

    - 답변이 적절한 경우 ➡️ 레시피 추천 LLM 답변 출력합니다.

    - 답변이 적절하지 않은 경우 ➡️ RAG 실행합니다.

                                
④ RAG 

    - 멀티 쿼리를 생성하여 문서 검색 성능 향상시킵니다.

    - 관련 문서가 있을 경우 ➡️ 내부문서 RAG 기반 답변 생성합니다.
                
    - 관련 문서가 없을 경우 ➡️ 외부문서 RAG 기반(Tavily) 답변 생성합니다.

                                
⑤ 최종 출력

    - 최종 생성된 답변 결과를 사용자에게 출력합니다.
""")


    st.markdown("""📌 이 프로세스를 통해 단순한 레시피 검색을 넘어서, 사용자의 상황에 최적화된 실행 가능한 요리 제안을 자동 생성할 수 있습니다.
    """)
                
if __name__ == "__main__":
    main()
