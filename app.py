import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import traceback

# Constants
EMBEDDING_MODEL = "embed-english-v3.0"
CHAT_MODEL = "command"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Set page config
st.set_page_config(
    page_title="Multi PDF Chatbot",
    page_icon=":scroll:",
    layout="wide"
)

def load_api_key():
    """Load and validate the Cohere API key."""
    load_dotenv("key.env")
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        st.error("Cohere API key not found. Please check your key.env file.")
        st.stop()
    return api_key

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                except UnicodeEncodeError:
                    continue
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
            continue
    return text

def get_text_chunks(text):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    """Create and save vector store from text chunks."""
    try:
        embeddings = CohereEmbeddings(
            model=EMBEDDING_MODEL,
            cohere_api_key=api_key,
            user_agent="langchain"
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def process_user_question(user_question, api_key):
    """Process user question and generate response."""
    try:
        # Load embeddings and vector store
        embeddings = CohereEmbeddings(
            model=EMBEDDING_MODEL,
            cohere_api_key=api_key,
            user_agent="langchain"
        )
        
        try:
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except FileNotFoundError:
            st.error("Please upload and process PDF files first.")
            return
        
        # Get relevant documents
        docs = new_db.similarity_search(user_question)
        doc_contents = "\n\n".join([doc.page_content for doc in docs])
        
        # Initialize LLM
        llm = Cohere(cohere_api_key=api_key, model=CHAT_MODEL)
        
        # Create and format prompt
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details and then add some additional points, if the answer is not in
        provided context, don't provide the wrong answer.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        formatted_prompt = prompt_template.format(context=doc_contents, question=user_question)
        
        # Generate and display response
        response = llm.invoke(formatted_prompt)
        st.write("Reply: ", response)
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    """Main application function."""
    # Load API key
    api_key = load_api_key()
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.header("Multi-PDF's Chatbot 🤖")
    
    # Main chat interface
    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️🗒")
    if user_question:
        process_user_question(user_question, api_key)
    
    # Sidebar
    with st.sidebar:
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files & Click on the Submit & Process Button",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
            
            with st.spinner("Processing..."):
                try:
                    # Process PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the PDF files.")
                        return
                    
                    # Create vector store
                    text_chunks = get_text_chunks(raw_text)
                    if get_vector_store(text_chunks, api_key):
                        st.success("PDFs processed successfully!")
                        
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")

if __name__ == "__main__":
    main()
