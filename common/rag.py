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
        - List all ingredients with accurate amounts (e.g., grams, ml, 개).
        - List all required cooking tools.
        - Provide step-by-step cooking instructions in numbered format (1, 2, 3, ...).
        - Clearly state the cooking time and tools used in each step.
        - Use traditional Korean cooking techniques (e.g., stir-frying, simmering, steaming).
        - Ensure the instructions are very detailed and easy to follow.

        ### Example Format:
        요리명: 김치볶음밥\n

        간단 설명: 남은 김치와 밥으로 간편하게 만드는 대표적인 한식 볶음밥입니다.\n

        필요한 재료:
        - 김치 100g
        - 밥 1공기
        - 대파 1/2대
        - 식용유 1큰술
        - 간장 1작은술
        \n
        필요한 조리도구:
        - 프라이팬
        - 주걱
        \n
        조리 순서:
        1. 프라이팬에 식용유 1큰술을 두르고 중불로 달군 후, 송송 썬 대파를 넣고 1분간 볶아 파기름을 냅니다.
        2. 김치 100g을 넣고 2분간 더 볶아줍니다.
        3. 밥 1공기를 넣고 간장 1작은술을 둘러 3분간 볶습니다.
        4. 불을 끄고 접시에 담아 완성합니다.

        **답변은 반드시 한글로 작성해 주세요.**
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
        ("human", "문맥(Context): {context}\n\n질문: {question}\
            위 질문에 대해 **요리명을 먼저 얘기하고, 도구, 조리 시간, 재료를 단계별로 포함하여 한국어로 작성**해 주세요.\
            반드시 요리순서나 요리과정에만 번호를 붙여서 구체적으로 알려주세요.\
            요리명, 도구, 조리 시간, 재료, 요리순서를 종합적으로 잘 정리해서 구체적으로 답변해주세요."
        )
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