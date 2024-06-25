import streamlit as st 
import os 
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Get the api keys in enviroment
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

# Setup LLM and embeddings model
llm = ChatGroq(model='llama3-8b-8192', temperature=0)
embedding_model = CohereEmbeddings()

# Write a function to save uploaded pdf as a temp file
def save_uploaded_pdf(uploaded_file):
    with open('temp.pdf', 'wb') as file:
        file.write(uploaded_file.getvalue())

def load_pdf_and_vectorstore(path='temp.pdf'):
    loader = PyPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(
        splits,
        embedding=embedding_model
    )
    vectorstore.save_local("faiss_index")

# Function to get response from model
def get_response(retriver, query):
    # Get system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
    ) 
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]     
    )
    # create qa chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # Create a retireval chain
    rag_chain = create_retrieval_chain(retriver, question_answer_chain)
    # Get response
    response = rag_chain.invoke({"input":query})
    return response["answer"]

# Streamlit application
def application():
    st.set_page_config(page_title="Chat with pdf", page_icon="💬")
    st.title("Chat with pdf - Utkarsh Gaikwad")
    st.subheader("Please upload the pdf file and get embeddings first")

    # Sidebar
    with st.sidebar:
        uploaded_file = st.file_uploader(
            label="Please upload pdf file here",
            type=['pdf'],
            accept_multiple_files=False
        )
        if uploaded_file is not None:
            if st.button("Get Embeddings"):
                with st.spinner("processing..."):
                    save_uploaded_pdf(uploaded_file)
                    load_pdf_and_vectorstore()
                    st.success('Done')
    
    # Main Window
    query = st.text_input("Please ask question related to uploaded pdf : ")
    if st.button("Answer"):
        with st.spinner("responding..."):
            db = FAISS.load_local(
                "faiss_index", 
                embedding_model,
                allow_dangerous_deserialization=True
            )
            retriever = db.as_retriever()
            response = get_response(retriever, query)
            st.write(response)

# Launch the streamlit application 
if __name__ == '__main__':
    application()