import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
)


chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")


qa = ConversationalRetrievalChain.from_llm(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )  

chat_history = []

res = qa({"question": "What are the applications of generative AI according the the paper? Please number each application.", "chat_history": chat_history})
print(res)

history = (res["question"], res["answer"])
chat_history.append(history)

res = qa({"question": "Can you please elaborate more on application number 2?", "chat_history": chat_history})
print(res)

# res = qa.invoke("What are the applications of generative AI according the the paper? Please number each application.")
# print(res) 

# res = qa.invoke("Can you please elaborate more on application number 2?")
# print(res)

# res = qa.invoke("Write everything you know about GenAI usage and Fashion and retail. Which paper is cited for this application?")
# print(res)