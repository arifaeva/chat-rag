from huey.contrib.djhuey import task

from core.ai.prompt_manager import PromptManager
from core.methods import send_chat_message
from core.ai.chromadb import chroma, openai_ef
from chats.models import Chat

import json

SYSTEM_PROMPT_RAG = """
Your are a helpful assistant.
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

    pm = PromptManager()
    pm.add_message("system", SYSTEM_PROMPT_RAG.format(documents=json.dumps(res)))
    pm.add_message("user", message)

    assistant_message = pm.generate()
    Chat.objects.create(role="assistant", content=assistant_message, document_id=document_id)

    send_chat_message(assistant_message)