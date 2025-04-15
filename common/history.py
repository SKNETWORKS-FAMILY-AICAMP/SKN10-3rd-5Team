
def add_history(message_history, role, content):
  message_history.append({"role": role, "content": content})

  return message_history
