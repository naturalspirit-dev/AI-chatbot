from flask import Flask, request, render_template
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.base_language import BaseLanguageModel
from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate



# env loading
from dotenv import load_dotenv
load_dotenv()
import os
# END env loading


app = Flask(__name__)

template = """Our platform offers two types of services: {services}.
At first you asked human which serivce you are going to receive.
Human will select the service name that they are going to recive in {services}.
Once human select service, please make appropriate questions from the below common questions for selected service.
Because we need to make requests about selected service in detail.

First of all, display the number of selected appropriate questions.
1. And then, ask to human one by one from start of approrpriate questions and their options for selected service to the end.
2. Human can select the options or input value instead of selection to answer appropriate question.
3. After human action, store that human input and ask next question in selected appropriate questions.

Repeate these 1,2,3 operations till the end of the appropriate questions.But you must don't ask same questions again.
If you asked all the appropriate questions, show all the human input values.
and say goodbye

Common Questions are these

1. What is your age range?

Less than 18
18 to 22
23 to 30
31 to 40
41 to 50
51 to 60
More than 60

2. How frequently would you like to train?

1 time a week
2 times a week
3 times a week
More than 3 times a week
I'm unsure, I would like some advice from the Personal Trainer

3. Where would you prefer to exercise?

In the gym
At my home
Outdoors
In a place recommended by the Personal Trainer
Online

4.How many individuals will the training involve?

Individual training
Pair training
Group training

5. When would you prefer to train? (You can select more than one time slot)

Early morning before 9 AM
Morning between 9 AM and 12 PM
Early afternoon between 1 PM and 3 PM
Late afternoon between 3 PM and 6 PM
Evening after 6 PM

6. What are your fitness goals?

Keep fit and tone up
Lose weight
Increase muscle mass
Improve performance
Post traumatic recovery
Improve posture
Other (please specify)

7.What is your gender?

Male
Female

8. Do you have a preference for the gender of your personal trainer?

No preference
Male
Female

9. How often do you currently exercise?

I don't engage in physical activity
1 or 2 times a week
More than 3 times a week

10. When do you require the service to start?

Within 2 days
Within 7 days
Within 15 days
In 30 days
I don't have an exact date

11. What is your email address?

12. Do you have any health issues we should be aware of?

Asthma
Back problems
Joint problems
Other (please specify)

13.What type of property requires cleaning?
Bungalow
Commercial property
Flat or Apartment
House
Other (please specify)

14. How frequently do you require cleaning services?
Daily
Twice a week
Weekly
Every other week
Once a month
One-time clean
Other (please specify)

15. How many bedrooms require cleaning?
0 bedrooms
1 bedroom
2 bedrooms
3 bedrooms
4 bedrooms
5+ bedrooms
Studio
Other (please specify)

16. How many bathrooms require cleaning?
1 bathroom
1 bathroom + 1 additional toilet
2 bathrooms
3 bathrooms
4+bathrooms
Other (please specify)

17. How many reception rooms require cleaning? (This includes lounge or dining rooms)
0
1
2
3
4+
Other (please specify)

18. What type of cleaning service are you looking for?
Standard cleaning
Deep cleaning
Move-out cleaning
Other (please specify)

19. What are the best days for the cleaning service?
Any
Monday
Tuesday
Wednesday
Thursday
Friday
Saturday
Sunday
I don't know
Other (please specify)

20. Will you be providing the cleaning materials and equipment?
Yes
No
Other (please specify)
How ready are you to hire a service?
I'm ready to hire now
I'm definitely going to hire someone
I'm likely to hire someone
I'm planning and researching
Other (please specify)

21.Where do you require the cleaning service? (Please provide the postcode or town for the address where you want the Cleaner)

End of Common Questions


Example Interaction:
Human: Let's start
AI: Hi there! Which service you are interested in - {services}?
Human:  Personal Training
AI:  To get that service you have to answer 10 appropriate questions. What is your age range?
Less than 18
18 to 22
23 to 30
31 to 40
41 to 50
51 to 60
More than 60
Human: {human_input}
{chat_history}
Chatbot:"""

llm = OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))
prompt = PromptTemplate(input_variables=["chat_history", "human_input", "services"], template=template)
qa_chain = LLMChain(llm=llm, prompt=prompt)

loader = TextLoader(os.getenv('TRAINDATA'))
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

import pinecone

# initialize pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment=os.getenv('PINECONE_ENV'),  # next to api key in console
)

index_name = os.getenv('PINECONE_INDEX')


docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
# if you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), docsearch.as_retriever())

query = "I want get personal trainser service"
# docs = docsearch.similarity_search(query)

# chain = load_qa_chain(llm=OpenAI(temperature=0), chain_type="stuff")

# result=chain.run(input_documents=docs, question=query)
chat_history=[]
chat_history.append("let us start")
res=qa({"question": "hi", "chat_history": chat_history})
chat_history.append(res['answer'])


@app.route('/hello/', methods=['GET'])
def render():
    return render_template('hello.html')

@app.route('/general-chats/', methods=['POST'])
def chatgpt():
    user_input = request.form['user_input']
    # response = chat.predict_messages([HumanMessage(content=user_input)])
    response = qa_chain.run(human_input=user_input, chat_history=chat_history, services=["Personal training", "Home cleaning"])
    print(app)
    chat_history.append([user_input, response])
    return response

@app.route('/init-chats/', methods=['DELETE'])
def initchathistory():
    chat_history = []
    return chat_history

@app.route('/trained-chats/', methods=['POST'])
def chatbot():
    user_input = request.form['user_input']
    result=qa({"question": user_input, "chat_history": chat_history})
    chat_history.append((result['answer'], user_input))
    print(result)
    return result['answer']

if __name__ == '__main__':
    app.run(host=os.getenv('HOST'), port=os.getenv('PORT'), debug=True)