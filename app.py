import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# **1. API Key Configuration:**
os.environ["GOOGLE_API_KEY"] = "AIzaSyAeTPGirysrTVbGj0qwMh8Gs7DAO50BEOw"  # Replace with your actual API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# **2. PDF Text Extraction (with Unicode Encoding Handling):**
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.encode("utf-8", "ignore").decode("utf-8")
            except UnicodeEncodeError as e:
                print(f"Unicode error: {e} - skipping problematic text.")
                continue
    return text

# **3. Text Chunking for Better Performance:**
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

# **4. Vector Store Creation:**
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# **5. Conversational Chain for Q&A:**
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details and then add some additional points, if the answer is not in
    provided context, don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# **6. User Input Processing and Display:**
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

# **7. Streamlit Web Interface:**
def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")

    # Custom background CSS
    background_path = "img/robots.jpg"
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: 'background_path';
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
""", unsafe_allow_html=True)
      
    # st.markdown(page_bg_img, unsafe_allow_html=True)

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
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        st.write("---")

if __name__ == "__main__":
    main()
