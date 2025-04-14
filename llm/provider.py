import enum

class Provider:
    def __init__(self, provider_LLMs:enum.Enum) -> None:
        self.provider_LLMs = provider_LLMs.__members__                  # 제공자의 모델 리스트
    
    # 호출 함수
    def __call__(self, model_name, messages):
        pass