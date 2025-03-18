import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import traceback
import time

# Set page config as the first Streamlit command
st.set_page_config(page_title="Multi PDF Chatbot", page_icon=":scroll:")

# **1. API Key Configuration:**
load_dotenv("key.env")  # Load API key from env file
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please check your key.env file.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Google API: {str(e)}")
    st.stop()

# Use the stable models
embedding_model = "models/embedding-001"
chat_model = "gemini-1.5-pro"

# **2. PDF Text Extraction (with Unicode Encoding Handling):**
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except UnicodeEncodeError:
                continue
    return text

# **3. Text Chunking for Better Performance:**
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks

# **4. Vector Store Creation:**
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "quota" in error_msg:
            st.error("API limit reached. Please try again later.")
        else:
            st.error("Error processing files. Please try again.")
        return False

# **5. User Input Processing and Display:**
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        try:
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except FileNotFoundError:
            st.error("Please upload and process PDF files first.")
            return
        
        docs = new_db.similarity_search(user_question)
        
        # Create the LLM
        llm = ChatGoogleGenerativeAI(model=chat_model, temperature=0.3)
        
        # Prepare the context from documents
        doc_contents = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a prompt
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details and then add some additional points, if the answer is not in
        provided context, don't provide the wrong answer.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        
        # Format the prompt with the actual context and question
        formatted_prompt = prompt_template.format(context=doc_contents, question=user_question)
        
        # Generate response
        response = llm.invoke(formatted_prompt)
        
        # Display response
        st.write("Reply: ", response.content)
        
    except Exception as e:
        st.error("Error processing question. Please try again.")

# **6. Streamlit Web Interface:**
def main():
    # Custom background CSS
    background_path = "img/robots.jpg"
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url('{background_path}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
""", unsafe_allow_html=True)
    
    st.header("Multi-PDF's Chatbot 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️🗒")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.image("img/robots.jpg")
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files & \n Click on the Submit & Process Button ",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
                
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the PDF files.")
                        return
                    text_chunks = get_text_chunks(raw_text)
                    if get_vector_store(text_chunks):
                        st.success("Done!")
                except Exception as e:
                    st.error("Error processing files. Please try again.")

        st.write("---")

if __name__ == "__main__":
    main()
