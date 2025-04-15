from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.retrievers import TavilySearchAPIRetriever
import streamlit as st

# 원본 데이터프레임을 전역 변수로 저장
original_df = pd.read_csv("./etl/rag/dataset/recipe_data.csv")

# 벡터 스토어 초기화 및 리트리버 생성
def get_retriever():
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "./etl/rag/dataset/recipe_faiss", 
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever()

# retriever에서 검색 결과가 나온 후, 조리순서를 추가하는 함수
def add_cooking_steps(retrieved_docs):
    enhanced_docs = []
    for doc in retrieved_docs:
        # 조리순서가 이미 포함된 경우 중복 제거를 위해 조리순서 이전 부분만 추출
        content = doc.page_content
        if "조리순서:" in content:
            content = content.split("조리순서:")[0].strip()
        # 레시피ID 추출
        recipe_ID = None
        content_lines = content.split('\n')
        for line in content_lines:
            if "레시피ID" in line:
                recipe_ID = line.split(': ')[1].strip()
                break
        if recipe_ID:
            # 원본 데이터프레임에서 해당 레시피ID의 조리순서 찾기
            recipe_row = original_df[original_df['레시피ID'] == int(recipe_ID)]
            if not recipe_row.empty and '조리순서' in recipe_row.columns:
                cooking_steps = recipe_row['조리순서'].values[0]
                # 기존 문서 내용(조리순서 제외)에 조리순서 추가
                enhanced_content = content + f"\n조리순서: {cooking_steps}"
                doc.page_content = enhanced_content
            else:
                doc.page_content = content
        enhanced_docs.append(doc)
    return enhanced_docs

# 다중 쿼리 생성 프롬프트
def get_multi_query_chain(llm_gpt):
    multi_query_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI language model assistant.
            Your task is to generate three different versions of the given user question to retrieve relevant documents from a vector database.
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
            Your response should be a list of values separated by new lines, eg: `foo\nbar\nbaz\n`
            Answer in Korean."""),
        ("human", "{question}")
    ])
    
    multi_query_chain = (
        {"question": RunnablePassthrough()}
        | multi_query_prompt
        | llm_gpt
        | StrOutputParser()
    )
    return multi_query_chain

def filter_relevant_docs(docs, query):
    """LLM을 사용하여 검색된 문서의 관련성을 평가하고 필터링합니다."""
    if not docs:
        return []
    print("필터되기 전:", len(docs))
    # 필터링을 위한 LLM 초기화 (같은 모델 재사용 가능)
    # llm = ChatGroq(model_name="gemma2-9b-it")
    llm_gpt = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.5,
                max_tokens=300,
            )
    filtered_docs = []
    
    for doc in docs:
        # 관련성 평가를 위한 프롬프트
        relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 검색 결과의 관련성을 평가하는 AI 어시스턴트입니다.
            사용자의 질문과 검색된 레시피 문서가 주어집니다.
            문서가 사용자의 질문 또는 의도와 관련이 있는지 판단해야 합니다.
            관련성이 높으면 '참'만 답변하고, 관련성이 낮거나 없으면 '거짓'만 답변하세요."""),
            ("human", f"사용자 질문: {query}\n\n검색된 문서: {doc.page_content}\n\n이 문서는 사용자 질문과 관련이 있나요? '참' 또는 '거짓'으로만 답변하세요.")
        ])
        
        # 관련성 평가 실행
        try:
            response = relevance_prompt | llm_gpt | StrOutputParser()
            result = response.invoke({}).strip().lower()
            
            # '참'인 경우에만 필터링된 문서 목록에 추가
            if result == '참':
                filtered_docs.append(doc)
        except Exception as e:
            # 오류 발생시 기본적으로 포함 (필터링 실패하더라도 검색 결과는 제공)
            print(f"문서 관련성 평가 중 오류 발생: {e}")
    print("필터되기 후:", len(filtered_docs))
    return filtered_docs

# 다중 검색 실행
def retrieve_with_steps(multi_queries, original_query):
    retriever = get_retriever()
    all_docs = []
    seen_docs = set()
    
    # 각 쿼리에 대해 검색 실행
    for q in multi_queries:
        if q.strip():  # 비어있지 않은 쿼리에 대해서만 실행
            docs = retriever.invoke(q)
            for doc in docs[:1]:
                if doc.page_content not in seen_docs:
                    seen_docs.add(doc.page_content)
                    doc.page_content = doc.page_content + "\n출처: " + "https://www.10000recipe.com/recipe/" + doc.page_content.split("시피ID: ")[1] + "\n----------------------\n"
                    all_docs.append(doc)
    
    # 검색 결과에 조리순서 추가
    enhanced_docs = add_cooking_steps(all_docs)

    # LLM을 사용하여 관련성 평가 및 필터링
    enhanced_docs = filter_relevant_docs(enhanced_docs, original_query)

    # 내부 문서 검색 결과가 없는 경우 Tavily 검색 수행
    if len(enhanced_docs) == 0:
        st.toast("인터넷을 검색하는 중...", icon="👨‍🍳")
        print("내부 문서 검색 결과 없음: Tavily 외부 검색 실행")
        try:
            # Tavily 검색 리트리버 초기화
            tavily_retriever = TavilySearchAPIRetriever(k=3)
            external_docs = []
            
            # 다중 쿼리로 Tavily 검색 실행
            for q in multi_queries:
                if q.strip():
                    try:
                        docs = tavily_retriever.invoke(q)
                        for doc in docs:
                            if doc.page_content not in seen_docs:
                                seen_docs.add(doc.page_content)
                                content = doc.page_content
                                source = doc.metadata["source"]
                                doc.page_content = content + "\n출처: " + source + "\n----------------------\n"
                                external_docs.append(doc)
                    except Exception as e:
                        print(f"Tavily 검색 중 오류 발생: {e}")
            
            # 외부 문서도 관련성 필터링 적용
            enhanced_docs = filter_relevant_docs(external_docs, original_query)
            print(f"Tavily 검색 결과: {len(enhanced_docs)}개 문서 찾음")
        except Exception as e:
            print(f"Tavily 검색 설정 중 오류 발생: {e}")
    
    # 문서 내용을 하나의 문자열로 합치기
    context_text = "\n\n---\n\n".join([doc.page_content for doc in enhanced_docs])
    return context_text

# message_history를 langchain 형식으로 변환
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

# 대화 요약을 위한 함수 추가
def summarize_conversation(chat_history, llm):
    # 요약을 위한 프롬프트 작성
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 대화 내용을 명확하고 간결하게 요약하는 AI 어시스턴트입니다.
        제공된 대화 기록을 분석하여 다음 정보를 포함하는 요약을 작성하세요:
        1. 사용자가 찾고 있는 레시피 유형
        2. 사용자가 언급한 재료, 좋아하는 재료
        3. 사용자가 싫어하는 재료
        4. 특별한 제약 조건(조리 시간, 사용 가능한 도구 등)
        
        핵심적인 정보만 포함하여 200자 이내로 간결하게 요약하세요."""),
        ("human", "다음 대화 내용을 요약해주세요:\n{conversation}")
    ])
    
    # 대화 내용을 텍스트로 변환
    conversation_text = ""
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            conversation_text += f"사용자: {msg.content}\n"
    
    # 요약 실행
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    try:
        summary = summarize_chain.invoke({"conversation": conversation_text})
        return summary
    except Exception as e:
        print(f"대화 요약 중 오류 발생: {e}")
        return ""

# 전역 대화 히스토리 저장소 추가
chat_histories = {}
# 이전 검색 결과와 대화 요약을 저장할 딕셔너리
past_contexts = {}
conversation_summaries = {}

# 최신 대화 요약을 가져오는 함수
def get_updated_summary(input_dict):
    session_id = input_dict.get("session_id", "default")
    return conversation_summaries.get(session_id, "아직 대화 요약이 없습니다.")

# 레시피 RAG 체인 생성
def create_rag_chain(llm_model, cooking_time=None, cooking_tools=None):
    # LLM 초기화
    llm = llm_model
    llm_gpt = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=300
    )
    
    # 다중 쿼리 체인
    multi_query_chain = get_multi_query_chain(llm_gpt)
    
    # 조리 시간과 도구를 포함한 프롬프트 템플릿
    system_message = """
        You are tasked with creating a recipe and detailed cooking instructions based on the ingredients provided.
        The recipe should:

        - Recommend a specific traditional Korean dish that uses the given ingredients.
        - Include a brief introduction to the dish.
        - List all ingredients with accurate amounts (e.g., grams, ml, pieces).
        - List all required cooking tools.
        - Provide step-by-step cooking instructions in numbered format (1, 2, 3, ...).
        - Clearly state the cooking time and tools used in each step.
        - Use traditional Korean cooking techniques (e.g., stir-frying, simmering, steaming).
        - Ensure the instructions are very detailed and easy to follow.
        - Utilize information from the previous context (past_contexts) in the response.
        - Refer to the conversation summary (conversation_summary) to understand the user's overall requirements and context.

        **# Example Format**:
        요리명: [요리명]
        간단 설명: [간단한 요리 설명]
        필요한 재료:
        - [재료 1]
        - [재료 2]

        필요한 조리도구:
        - [조리 도구 1]
        - [조리 도구 2]

        조리 순서:
        1. [조리 순서 1]
        2. [조리 순서 2]
        3. [조리 순서 3]
        4. [조리 순서 4]
        
        출처:

        **Please ensure your response strictly follows the Example Format. Additionally, the response must be in Korean.**
        **반드시 주어진 문서의 출처를 포함하세요**
    """

    # 조리 시간 제약 추가
    if cooking_time:
        system_message += f"\n#조리 시간: {cooking_time}"
    
    # 조리 도구 제약 추가
    if cooking_tools and len(cooking_tools) > 0:
        system_message += f"\n#사용 가능한 조리 도구: {', '.join(cooking_tools)}"
    
    # 이전 컨텍스트와 대화 요약을 프롬프트에 포함
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("human", """
        이전 대화 요약(conversation_summary): {conversation_summary}\n
        이전 문맥(past_contexts): {past_contexts}\n
        현재 문맥(Context): {context}\n
        질문: {question}\n
        
        위 질문에 대해 **반드시 조리순서, 도구, 조리 시간, 재료를 단계별로 포함하여 한국어로 작성**해 주세요.
        """)
    ])
    
    # 검색 함수 정의
    def retrieve_context(input_dict):
        session_id = input_dict.get("session_id", "default")
        current_query = input_dict["question"]
        chat_history = input_dict.get("chat_history", [])
        # 임시 대화 기록에 현재 질문 추가
        temp_chat_history = chat_history.copy()
        temp_chat_history.append(HumanMessage(content=current_query))
        
        # 대화 요약 생성/업데이트 (현재 질문 포함)
        if len(temp_chat_history) > 0:
            conversation_summaries[session_id] = summarize_conversation(temp_chat_history, llm)
        print("conversation_summaries:", conversation_summaries)
        # 대화 기록에서 사용자 질문 추출
        user_questions = []
        
        # 대화 기록에서 사용자 메시지 추출
        for message in chat_history:
            if isinstance(message, HumanMessage):
                user_questions.append(message.content)
        
        # 현재 질문 추가
        user_questions.append(current_query)
        
        # 모든 사용자 질문을 하나의 문자열로 결합
        combined_query = "\n- ".join(user_questions)
        
        # 다중 쿼리 생성 후 각각에 대해 검색 실행
        multi_queries = multi_query_chain.invoke(current_query).strip().split("\n")
        
        # 검색 결과 가져오기
        current_context = retrieve_with_steps(multi_queries, combined_query)
        
        # 세션별 이전 컨텍스트 업데이트
        if session_id not in past_contexts:
            past_contexts[session_id] = []
        
        # 현재 컨텍스트를 이전 컨텍스트 목록에 추가 (너무 길어지지 않게 최근 2개만 유지)
        past_contexts[session_id].append(current_context)
        if len(past_contexts[session_id]) > 2:
            past_contexts[session_id] = past_contexts[session_id][-2:]
        
        return current_context
    
    # 기본 RAG 체인 구성
    base_chain = (
        {
            "context": RunnableLambda(retrieve_context),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "past_contexts": lambda x: "\n\n===PREVIOUS CONTEXT===\n\n".join(
                past_contexts.get(x.get("session_id", "default"), [])
            ),
            "conversation_summary": RunnableLambda(get_updated_summary)
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 세션에 따른 대화 히스토리 가져오기 함수
    def get_chat_history(session_id):
        if session_id not in chat_histories:
            chat_histories[session_id] = ChatMessageHistory()
        return chat_histories[session_id]
    
    # 대화 기록을 관리하는 체인으로 변환
    rag_chain = RunnableWithMessageHistory(
        base_chain,
        get_chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    return rag_chain