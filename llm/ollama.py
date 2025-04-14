# 필요한 모듈들을 임포트
import enum
from langchain_community.chat_models import ChatOllama

from llm.provider import Provider

# Ollama에서 사용할 수 있는 LLM 모델들을 정의하는 열거형 클래스
class OLLAMA_LLMs(enum.Enum):
    gemma3_4b_q8 = (enum.auto(), "gemma3-q8")
    gemma3_4b_q8_recipe = (enum.auto(), "gemma3-recipe")
    
class Provider_Ollama(Provider):
    def __init__(self, provider_LLMs=OLLAMA_LLMs):
        super().__init__(provider_LLMs)
    
    # Ollama 모델을 호출하는 메서드
    def __call__(self, model_name):
        # 선택된 모델로 ChatOllama 인스턴스 생성
        llm = ChatOllama(
            model=self.provider_LLMs[model_name].value[1],
            model_kwargs={
                            "max_tokens": 1000,
                            "temperature": 0.7
                        }
        )
        return llm