
from pymongo import MongoClient
from langchain_groq import ChatGroq
import os
import getpass
from datetime import datetime

db=MongoClient()
qa_chatbot=db['qa_chatbot']
qa_collection=qa_chatbot['qa_collection']

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key='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'

)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence."
    ),
    ("human", "I love programming,mathematics,physics,english and python."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)
print(messages[1][1])
qa_collection.insert_one(

    {'question':messages[1][1],
    'answer':ai_msg.content,
    'date':datetime.now()
    }

    )
