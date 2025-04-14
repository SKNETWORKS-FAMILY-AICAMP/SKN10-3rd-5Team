
import time
import streamlit as st

from common.history import add_history
from common.display import write_chat
from common.rag import create_rag_chain
from llm.ollama import Provider_Ollama


def get_llm_model():
    # Provider_Ollama 인스턴스 생성
    provider = Provider_Ollama()

    # 'gemma3_4b_q8_recipe' 모델을 선택하여 ChatOllama 인스턴스를 반환
    return provider("gemma3_4b_q8_recipe")


def get_response_from_llm(message_history, cooking_time, cooking_tools, session_id="default"):
    user_message = message_history[-1]["content"]

    # RAG 체인을 사용하여 레시피 답변 생성
    rag_chain = create_rag_chain(get_llm_model(), cooking_time, cooking_tools)

    # 스트리밍 응답 생성
    for token in rag_chain.stream(
        {"question": user_message},
        config={"configurable": {"session_id": session_id}},
    ):
        yield token
        time.sleep(0.05)

    # client = OpenAI()
    # response = client.chat.completions.create(
    #   model="gpt-4o-mini",
    #   messages=message_history,
    #   stream=True,
    # )

    # for token in response:
    #   if token.choices[0].delta.content is not None:
    #     yield token.choices[0].delta.content
    #     time.sleep(0.05)

def ask(question, message_history, cooking_time=None, cooking_tools=None):
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
        message=get_response_from_llm(message_history, cooking_time, cooking_tools, st.session_state.session_id)  # 세션 ID 전달
    )
    message_history = add_history(message_history, role="assistant", content=response)

    return message_history
