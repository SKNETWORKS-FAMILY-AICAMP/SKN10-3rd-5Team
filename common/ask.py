import time
import streamlit as st

from common.history import add_history
from common.display import write_chat
from common.rag import create_rag_chain
from llm.ollama import Provider_Ollama

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Groq LLM 초기화
groq_llm = ChatGroq(model_name="qwen-2.5-32b")  # 너가 쓰는 모델로 변경 가능

def is_cooking_related_question_groq(user_input):
    system_prompt = "이 질문이 요리 레시피로 무엇을 만들 수 있냐는 질문이라면 '요리', 아니면 '일반'이라고만 정확하게 한 단어로 대답해줘. 반드시 한국어로."
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


def get_response_from_llm(message_history, cooking_time, cooking_tools, session_id="default", llm_model=None):
  user_message = message_history[-1]["content"]

  if is_cooking_related_question_groq(user_message):
    # RAG 체인을 사용하여 레시피 답변 생성
    if llm_model is None:
        llm_model = get_llm_model()
    rag_chain = create_rag_chain(llm_model, cooking_time, cooking_tools)

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
        You are a helpful cooking recipe assistant. Based on the provided ingredients, you should provide detailed recipes and cooking steps in an organized manner.
        Your response must be in Korean, and you should list the steps in numbered order.
        
        1. Recipe Name: Clearly state the name of the dish.
        2. Ingredients: List the ingredients needed for the dish, including the exact amounts for each.
        3. Tools: List the cooking tools that should be used in the process, such as frying pans, mixing bowls, etc.
        4. Cooking Time: Specify the estimated cooking time for each step.
        5. Cooking Steps: Provide the cooking steps, numbered and explained clearly. For example, '1. Prepare the ingredients' or '2. Stir-fry for 5 minutes'.
        
        Include all of this information to ensure that the user can easily follow the recipe and successfully recreate the dish.
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
