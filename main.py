
from pymongo import MongoClient
from langchain_groq import ChatGroq
import os
import getpass
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


client=MongoClient()
qa_chatbot=client['qa_chatbot']
qa_collection=qa_chatbot['qa_collection']


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


question_answer=[messages[1][1],ai_msg.content]  #saving  the question into the mongodb

model_name = "sentence-transformers/all-MiniLM-L6-v2" #initializing the model name

embedding_model = HuggingFaceEmbeddings(model_name=model_name)  #making the embeddings of models using the Hugging Face

embeddings = embedding_model.embed_documents(question_answer)

qa_collection.insert_one({    #inserting the question and ai response in the mongodb with their embeddings
    'question':messages[1][1],
    'answer':ai_msg.content,

    
        'question_embeddings':embeddings[0],
        'answer_embeddings':embeddings[1],
            'date':datetime.now()
    })


user_input=input('enter the message ') 

embedding_model = HuggingFaceEmbeddings(model_name=model_name)


input_embeddings = embedding_model.embed_query(user_input)
user_embeddings=np.array(input_embeddings)  #converting the user query embeddings into 2D array
user_embeddings=user_embeddings.reshape(1,-1) 

documents=list(qa_collection.find()) #finding all the embeddings of documents from mongodb

best_answer = ""
similarity_list=[] 
for document in documents:

    db_embeddings=document['answer_embeddings'] #getting the individual answer embeddings from the document 
    db_embeddings=np.array(db_embeddings) #converting the user query embeddings into 2D array
    db_embeddings=db_embeddings.reshape(1,-1)

    similarity = cosine_similarity(user_embeddings, db_embeddings)[0][0] #finding the similarity score
    similarity_list.append(similarity) #inserting the individual score into the list in order to find the maximum score 


max_similarity=max(similarity_list) #finding the maximum similarity

max_value_index=similarity_list.index(max(similarity_list))  #finding the index of the  maximum similarity
best_answer=documents[max_value_index]['answer']  #from that index we find the coressponding answer into the mongo db

print(f"output: {best_answer}")
print(f"Similarity score: {max_similarity}")
