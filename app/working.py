from flask import Flask, request, render_template
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.chains import ConversationChain
from langchain.llms import OpenAI

from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings


# env loading
from dotenv import load_dotenv
load_dotenv()
import os
# END env loading


app = Flask(__name__)

services=['Personal Trainer', 'Home cleaning']
formatted_services = '\n- '.join(services)
formatted_list = f'- {formatted_services}'

with open(os.getenv('TRAINDATA'), 'r') as file:
    text = file.read()

template = f"""Our platform offers two types of services: {', '.join(services)}.
At first you must ask to human which serivce is interested in.
Wait util human input.
After that human will select the service name that they are going to recive one of services like {', '.join(services)}.
Once human select service, please make appropriate questions from the below common questions for selected service and it must be less than 11
Because we need to make requests about selected service in detail.

First of all, display the number of selected appropriate questions.
1. And then, ask to human one by one from start of approrpriate questions and their options for selected service to the end.
2. Human can select the options or input value instead of selection to answer appropriate question.
3. After human action, store that human input and ask next question in selected appropriate questions.

Repeate these 1,2,3 operations till the end of the appropriate questions.But you must don't ask same questions again.
If you asked all the appropriate questions, show all the human input values.
And say goodbye and display ***END*** at last.


{text}


Example Conversation:
AI: Hi there! Which service are you interested? We offer {', '.join(services)}
Human: Personal Trainer
AI: To get that service, you have to answer 10 appropriate questions. What is your age range?
   - Less than 18
   - 18 to 22
   - 23 to 30
   - 31 to 40
   - 41 to 50
   - 51 to 60
   - More than 60
Human: 23 to 30.
AI: What is your gender?
   - Male
   - Female
Human: Male.
...

So let's start conversation now."""


llm = OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))

# We set a low k=100, to only keep the last 100 interactions in memory
memory=ConversationBufferWindowMemory( k=10, return_messages=True, )

conversation_with_summary = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=False
)

# loader = TextLoader(os.getenv('TRAINDATA'))
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# embeddings.embed_documents(docs)
@app.route('/hello/', methods=['GET'])
def render():
    response = conversation_with_summary.predict(input = template)
    memory.save_context({"input": template}, {"output": response})
    return render_template('hello.html', result=response)

@app.route('/general-chats/', methods=['POST'])
def chatgpt():
    user_input = request.form['user_input']
    response = conversation_with_summary.predict(input = user_input)
    memory.save_context({"input": user_input}, {"output": response})

    return response

@app.route('/init-chats/', methods=['DELETE'])
def initchathistory():
    memory.clear()
    conversation_with_summary.memory = memory
    response = conversation_with_summary.predict(input = template)
    memory.save_context({"input": template}, {"output": response})

    return response

if __name__ == '__main__':
    app.run(host=os.getenv('HOST'), port=os.getenv('PORT'), debug=True)