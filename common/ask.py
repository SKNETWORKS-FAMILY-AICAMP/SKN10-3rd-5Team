import time
import streamlit as st

from common.history import add_history
from common.display import write_chat
from common.rag import create_rag_chain
from llm.ollama import Provider_Ollama

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Groq LLM 초기화
groq_llm = ChatGroq(model_name="gemma2-9b-it")  # 너가 쓰는 모델로 변경 가능

def is_cooking_related_question_groq(user_input):
    # system_prompt = "이 질문이 요리 레시피로 무엇을 만들 수 있냐는 질문이라면 '요리', 아니면 '일반'이라고만 정확하게 한 단어로 대답해줘. 반드시 한국어로."
    system_prompt = """
      이 질문이 '요리 레시피', '레시피', 또는 '음식'과 관련된 질문이라면 '요리'라고만 정확하게 한 단어로 대답해 주세요.
      만약 요리와 관련이 없다면 '일반'이라고만 정확하게 한 단어로 대답해 주세요.
      예시: '달걀과 양파를 활용한 레시피를 추천해줘.' -> '요리'
      """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    response = groq_llm.invoke(messages).content.strip()
    return response == "요리"

def get_llm_model():
    # Provider_Ollama 인스턴스 생성
    provider = Provider_Ollama()

    # 'gemma3_4b_q8_recipe' 모델을 선택하여 ChatOllama 인스턴스를 반환
    return provider("gemma3_4b_q8_recipe")

def convert_messages(message_history):
    chat_history = []
    for msg in message_history:
        if msg["role"] == "system":
            chat_history.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history

def filter_relevant_respond(respond, query):
    """LLM을 사용하여 검색된 문서의 관련성을 평가하고 필터링합니다."""
    # 필터링을 위한 LLM 초기화 (같은 모델 재사용 가능)
    
    llm_gpt = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.5,
                max_tokens=300,
              )
    # 관련성 평가를 위한 프롬프트
    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 검색 결과의 관련성을 평가하는 AI 어시스턴트입니다.
        생성된 답변이 사용자의 질문에 적절한 답변인지 판단해야 합니다.
        또한, 답변은 반드시 **예시 형식**에 맞게 작성되어야 합니다:
        
        **예시 형식**:
        요리명: [요리명]
        간단 설명: [간단한 요리 설명]
        필요한 재료:
        - [재료 1]
        - [재료 2]
        필요한 조리 도구:
        - [조리 도구 1]
        - [조리 도구 2]
        조리 순서:
        1. [조리 순서 1]
        2. [조리 순서 2]
        3. [조리 순서 3]
        4. [조리 순서 4]
        
        만약 생성된 답변이 이 **예시 형식**에 맞고 사용자의 질문과 관련이 있다면 '참'을 답변하세요.
        만약 생성된 답변이 형식에 맞지 않거나 질문과 관련이 없다면 '거짓'만 답변하세요.
        """),
        ("human", f"사용자 질문: {query}\n\n생성된 답변: {respond}\n\n이 답변은 예시 형식에 맞고 사용자 질문과 관련이 있나요? '참' 또는 '거짓'으로만 답변하세요.")
    ])
    
    # 관련성 평가 실행
    try:
        response = relevance_prompt | llm_gpt | StrOutputParser()
        result = response.invoke({}).strip().lower()
        
        # '참'인 경우에만 필터링된 문서 목록에 추가
        if result == '참':
            return True
        else:
            return False
    except Exception as e:
        # 오류 발생시 기본적으로 포함 (필터링 실패하더라도 검색 결과는 제공)
        print(f"문서 관련성 평가 중 오류 발생: {e}")
        return False


def get_response_from_llm(message_history, cooking_time, cooking_tools, session_id="default", llm_model=None):
  user_message = message_history[-1]["content"]

  if is_cooking_related_question_groq(user_message):
    # RAG 체인을 사용하여 레시피 답변 생성
    if llm_model is None:
      llm_model = get_llm_model()
    
    # 대화 히스토리를 프롬프트 형식으로 변환
    prompts = []
    
    for msg in message_history[:-1]:                   # 사용자의 메시지는 제외해야 함.
      print(msg)
      prompts.append(tuple(msg.values()))
        
    # 마지막 사용자 입력을 위한 프롬프트 추가
    prompts += [("user", "{user_input}")]
    
    print("prompts", prompts)
    
    # 채팅 프롬프트 템플릿 생성
    chat_prompt = ChatPromptTemplate.from_messages(prompts)

    # 프롬프트 -> LLM -> 문자열 파서로 이어지는 체인 생성
    chain = chat_prompt | llm_model | StrOutputParser()
    
    # LLM의 전체 응답을 받음
    response = chain.invoke({"user_input": user_message}).strip()
    print("LLM 응답:", response)
    # 사용자 질문과 답변의 관련성 평가
    is_relevant = filter_relevant_respond(response, user_message)
    print("답변 관련성 평가:", is_relevant)
    # 답변이 관련성이 있다고 판단되면 스트림처럼 출력
    if is_relevant:
      # 스트리밍 효과: 답변을 한 글자씩 출력하여 "실시간" 효과를 준다
      for char in response:
        yield char
        time.sleep(0.05)  # 약간의 지연을 주어 스트리밍 효과를 준다
    else:
      rag_chain = create_rag_chain(groq_llm, cooking_time, cooking_tools)

      # 스트리밍 응답 생성
      for token in rag_chain.stream(
        {"question": user_message},
        config={"configurable": {"session_id": session_id}},
      ):
        yield token
        time.sleep(0.05)

  else:
    # ✅ 일반 질문일 경우 → Groq GPT 직접 응답
    messages = [SystemMessage(content="친절한 요리 AI 어시스턴트입니다. 반드시 한국어로 답하세요.")] + [
      HumanMessage(content=msg["content"]) if msg["role"] == "user" else
      SystemMessage(content=msg["content"]) if msg["role"] == "system" else
      HumanMessage(content=msg["content"])  # assistant도 HumanMessage처럼 처리
      for msg in message_history if msg["role"] != "system"
    ]

    for chunk in groq_llm.stream(messages):
      if hasattr(chunk, "content") and chunk.content:
        yield chunk.content
        time.sleep(0.05)

def ask(question, message_history, cooking_time=None, cooking_tools=None, llm_model=None):
  if len(message_history) == 0:
    # 최초 시스템 프롬프트
    message_history.append({
        "role": "system", 
        "content": """
        당신은 요리 레시피와 조리 방법을 제공하는 AI입니다. 주어진 재료를 바탕으로 정확한 형식에 맞는 레시피와 조리 방법을 제공합니다. 
        반드시 다음 형식에 맞게 답변해주세요:

        #Example Format:
        (간단한 요리 설명)

        | 항목          | 내용                                      |
        | ------------- | ----------------------------------------- |
        | **요리명**    | [요리명 1]                                  |
        | **간단 설명** | [간단한 요리 설명 1]                        |
        | **필요한 재료** | [재료 1], [재료 2], [재료 3] (예: 그램, ml, 개 단위) |
        | **필요한 조리 도구** | [조리 도구 1], [조리 도구 2]              |
        | **조리 순서** | 1. [조리 순서 1] <br> 2. [조리 순서 2] <br> 3. [조리 순서 3] <br> 4. [조리 순서 4] |

        **답변은 반드시 위의 예시 형식을 지켜야 하며, 모든 답변은 한글로 작성되어야 합니다. 줄글 형식은 절대 허용되지 않습니다.**
        """
    })




  # 사용자 질문 추가 및 즉시 표시
  message_history = add_history(message_history, role="user", content=question)
  write_chat(role="user", message=message_history[-1]["content"])

  # 세션 ID 생성 (사용자마다 고유한 ID 사용)
  import uuid
  if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
  
  # LLM 답변 즉시 표시 및 추가
  response = write_chat(
    role="assistant",
    message=get_response_from_llm(message_history, cooking_time, cooking_tools, st.session_state.session_id, llm_model)  # 세션 ID 전달
  )
  message_history = add_history(message_history, role="assistant", content=response)

  return message_history
