from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain import OpenAI

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from flask import Flask


# env loading
from dotenv import load_dotenv
load_dotenv()
import os
# END env loading


llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))
loader = TextLoader("app/docs/bank.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)
chat_history = []
query = ""

app = Flask(__name__)

@app.route('/init-chats/', methods=['DELETE'])
def initchathistory():
    chat_history.clear()
    return 'Inited'

@app.route('/trained-chats/', methods=['POST'])
def chat():
    response = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, response["answer"]))
    return response['answer']


# ~~~~~~~~~~~~~~~~ START ~~~~~~~~~~~~~~~~

# sm_loader = UnstructuredFileLoader("app/docs/bank.txt")
# sm_doc = sm_loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=400,
#     chunk_overlap=0
# )
# lg_docs =text_splitter.split_documents(sm_doc)


# def doc_summary(docs) :
#     print(f'You have {len(docs)} document(s)')

#     num_words = sum([len(doc.page_content.split(' ')) for doc in docs])

#     print(f'You have roughly {num_words} words in your docs')
#     print()
#     print(f'Preview: \n{docs[0].page_content.split(". ")[0]}')

# doc_summary(sm_doc)
# doc_summary(lg_docs)

# chain = load_summarize_chain(llm, chain_type="refine")
# chain.run(lg_docs)

# ~~~~~~~~~~~~~~~~ END ~~~~~~~~~~~~~~~~
