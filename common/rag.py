from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
import pandas as pd
import os

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
def get_multi_query_chain(llm):
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
        | llm
        | StrOutputParser()
    )
    return multi_query_chain

# 다중 검색 실행
def retrieve_with_steps(multi_queries):
    retriever = get_retriever()
    all_docs = []
    seen_docs = set()
    
    # 각 쿼리에 대해 검색 실행
    for q in multi_queries:
        if q.strip():  # 비어있지 않은 쿼리에 대해서만 실행
            docs = retriever.invoke(q)
            for doc in docs:
                if doc.page_content not in seen_docs:
                    seen_docs.add(doc.page_content)
                    all_docs.append(doc)
    
    # 검색 결과에 조리순서 추가
    enhanced_docs = add_cooking_steps(all_docs)
    
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

# 전역 대화 히스토리 저장소 추가
chat_histories = {}

# 레시피 RAG 체인 생성
def create_rag_chain(cooking_time=None, cooking_tools=None):
    # LLM 초기화
    llm = ChatGroq(model_name="qwen-2.5-32b")
    
    # 다중 쿼리 체인
    multi_query_chain = get_multi_query_chain(llm)
    
    # 조리 시간과 도구를 포함한 프롬프트 템플릿
    system_message = """당신은 질문-답변(Question-Answering)을 수행하는 레시피 추천 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요.
문맥(context) 은 레시피에 대한 정보입니다. 주어진 질문(question)에 대해 적절한 레시피를 추천해주세요.
레시피에 대한 레시피 이름, 재료, 도구, 조리 순서를 답변(Answer) 에 포함하세요.
반드시 한글로 답변해 주세요.
"""
    
    # 조리 시간 제약 추가
    if cooking_time:
        system_message += f"\n#조리 시간: {cooking_time}"
    
    # 조리 도구 제약 추가
    if cooking_tools and len(cooking_tools) > 0:
        system_message += f"\n#사용 가능한 조리 도구: {', '.join(cooking_tools)}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("human", "문맥(Context): {context}\n\n질문: {question}")
    ])
    
    # 검색 함수 정의
    def retrieve_context(input_dict):
        query = input_dict["question"]
        # 다중 쿼리 생성 후 각각에 대해 검색 실행
        multi_queries = multi_query_chain.invoke(query).strip().split("\n")
        return retrieve_with_steps(multi_queries)
    
    # 기본 RAG 체인 구성
    base_chain = (
        {
            "context": RunnableLambda(retrieve_context),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
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
        get_chat_history,  # 수정된 부분
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    return rag_chain