
import time
from openai import OpenAI
from groq import Groq


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
