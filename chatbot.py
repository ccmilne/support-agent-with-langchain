import streamlit
import os
import re

#Context-specific
from langchain.document_loaders import PyPDFLoader

#Pipeline needs
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool

#Environment-specific
from langchain.embeddings import VertexAIEmbeddings
from langchain.chat_models import ChatVertexAI

### Env Vars
vertexai_api_key = os.getenv('VERTEXAI_API_KEY')

### Prompt Design
support_template = """As a Verizon customer service agent, your goal is to provide accurate and helpful information about Verizon products and telecommunications concepts.
You should answer customer questions based on the context provided and avoid making up answers. Remember to provide relevant information and be as concise as possible

Context: {context}
Question: {question}"""

SUPPORT_PROMPT = PromptTemplate(template=support_template, input_variables=["context", "question"])

sales_template = """As a Verizon sales representative, your goal is to sell Verizon products and services to customers.
You should answer customer questions based on the context while continually reminding them of products or services offered by Verizon that might respond to their question. 
Remember to provide relevant information and be as concise as possible

Context: {context}
Question: {question}"""

SALES_PROMPT = PromptTemplate(template=sales_template, input_variables=["context", "question"])

prompts = {'SUPPORT_PROMPT': SUPPORT_PROMPT, 'SALES_PROMPT': SALES_PROMPT}

def initialize_database():
    """Loads and chunks Verizon files into Chroma db
    """
    for file in os.listdir('data/pdfs/'):
        new_name = re.sub(" ", "_", file).lower()
        loaded = PyPDFLoader(file)
        splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        docs = splitter.split_documents(loaded)
        db = Chroma.from_documents(docs, VertexAIEmbeddings(), collection_name=new_name)

def initialize_agent(db, **kwargs):
    """_summary_

    Args:
        db (_object_): Chroma vector store

    Returns:
        agent (_object_): LangChain agent
    """
    support_chain = RetrievalQA.from_chain_type(
        llm=ChatVertexAI(),
        chain_type='stuff',
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": kwargs['SUPPORT_PROMPT']}
    )

    sales_chain = RetrievalQA.from_chain_type(
        llm=ChatVertexAI(temperature=0.7), #ramp up the creativity and schmooze
        chain_type='stuff',
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": kwargs['SALES_PROMPT']}
    )

    tools = [
        Tool(
            name = "Support",
            func = support_chain.run,
            description = """Useful for when a customer is interested in learning more about Verizon products, services, or other concepts,
                            needs help with understanding. Input should be a fully formed question"""
        ),
        Tool(
            name = "Sales",
            func = sales_chain.run,
            description = """Useful for when a customer indicates they are interested in a Verizon product or service, and that they are
                            willing to buy. Input should be a fully formed question."""
        )
    ]

    agent = initialize_agent(tools, ChatVertexAI(), agent="zero-shot-react-description", verbose=True)
    return agent


### Webpage Setup
streamlit.set_page_config(page_title="Verizon Customer Support", page_icon=":random:", layout='centered')
streamlit.title('Verizon Customer Support')

question = streamlit.text_area("Ask a question about your Verizon product", "", placeholder="Example: What will 5G mean for my current phone?")

### Conversation
with streamlit.form("answer_form", clear_on_submit=True):

    #Prep resources
    vectordb = initialize_database()
    agent = initialize_agent(vectordb, prompts)

    gcp_key = streamlit.text_input('OpenAI API Key', type='password', disabled=not question, value=streamlit.secrets['VERTEXAI_API_KEY'])
    submitted = streamlit.form_submit_button("SUBMIT")

    if submitted and gcp_key:
        with streamlit.spinner("Answering ..."):
            response = agent.run(question)
            if response:
                streamlit.write(response)