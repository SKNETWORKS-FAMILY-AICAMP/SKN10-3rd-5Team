
import time
from openai import OpenAI

from common.history import add_history
from common.display import write_chat


def get_response_from_llm(message_history):
  client = OpenAI()
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=message_history,
    stream=True,
  )

  for token in response:
    if token.choices[0].delta.content is not None:
      yield token.choices[0].delta.content
      time.sleep(0.05)

def ask(question, message_history):
  if len(message_history) == 0:
    # 최초 시스템 프롬프트
    message_history.append({"role": "system", "content": "You are a helpful assistant. You must answer in Korean."})

  # 사용자 질문 추가 및 즉시 표시
  message_history = add_history(message_history, role="user", content=question)
  write_chat(role="user", message=message_history[-1]["content"])

  # LLM 답변 즉시 표시 및 추가
  response = write_chat(role="assistant", message=get_response_from_llm(message_history))
  message_history = add_history(message_history, role="assistant", content=response)

  return message_history
