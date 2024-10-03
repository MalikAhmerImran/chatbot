
from pymongo import MongoClient
from langchain_groq import ChatGroq
import os
import getpass
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings

client=MongoClient()
qa_chatbot=client['qa_chatbot']
qa_collection=qa_chatbot['qa_collection']
db=client['embeddings']
embedding_collection=db['embedding_collection']

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=''

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

question_answer=[messages[1][1],ai_msg.content]

model_name = "sentence-transformers/all-MiniLM-L6-v2" 

embedding_model = HuggingFaceEmbeddings(model_name=model_name)

embeddings = embedding_model.embed_documents(question_answer)

print(embeddings)

embedding_collection.insert_one({
        'question_embeddings':embeddings[0],
        'answer_embeddings':embeddings[1]
    })