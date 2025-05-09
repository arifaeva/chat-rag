from huey.contrib.djhuey import task

from core.ai.prompt_manager import PromptManager
from core.methods import send_chat_message
from core.ai.chromadb import chroma, openai_ef
from core.ai.tokenizer import count_token
from chats.models import Chat

import json

SYSTEM_PROMPT_RAG = """
You are a helpful assistant.
Your task is to answer user question based on the provided document.

PROVIDED DOCUMENTS:
{documents}

ANSWER GUIDELINES:
- Always answer in bahasa indonesia.
- Do not include any additional information other than provided document.
"""

@task()
def process_chat(message, document_id):
    Chat.objects.create(role="user", content=message, document_id=document_id)

    collection = chroma.get_or_create_collection(document_id, embedding_function=openai_ef)
    res = collection.query(query_texts=[message], n_results=3)

    messages = []
    chats = Chat.objects.filter(document_id=document_id)

    for chat in chats:
        messages.append({"role": chat.role, "content": chat.content})

    system_prompt = SYSTEM_PROMPT_RAG.format(documents=json.dumps(res))
    system_prompt_token = count_token(system_prompt)

    pm = PromptManager()
    pm.add_message("system", system_prompt)
    pm.add_messages(messages=messages)

    messages_token = count_token(json.dumps(messages))

    assistant_message = pm.generate()
    assistant_token = count_token(assistant_message)

    print(system_prompt_token)
    print(messages_token)
    print(assistant_token)

    Chat.objects.create(role="assistant", content=assistant_message, document_id=document_id)

    send_chat_message(assistant_message)