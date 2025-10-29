import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import traceback

# Constants
EMBEDDING_MODEL = "embed-english-v3.0"
# Default chat model; can be overridden by COHERE_CHAT_MODEL env var
CHAT_MODEL = "command-a-03-2025"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Set page config
st.set_page_config(
    page_title="Multi PDF Chatbot",
    page_icon=":scroll:",
    layout="wide"
)

def load_api_key():
    load_dotenv("key.env")
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        st.error("Cohere API key not found. Please check your key.env file.")
        st.stop()
    return api_key

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except UnicodeEncodeError:
                    continue
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
            continue
    return text

def get_text_chunks(text):
    if not text.strip():
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    if not text_chunks:
        return False
    try:
        embeddings = CohereEmbeddings(
            model=EMBEDDING_MODEL,
            cohere_api_key=api_key
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def _invoke_with_fallback(api_key, preferred_model, prompt):
    # Try a sequence of known supported models in case of deprecations
    candidate_models = []
    if preferred_model:
        candidate_models.append(preferred_model)
    candidate_models.extend([
        "command-a-03-2025",
        "command-a"
    ])

    last_error = None
    for model_name in candidate_models:
        try:
            llm = ChatCohere(
                cohere_api_key=api_key,
                model=model_name,
                temperature=0.7,
                max_tokens=500
            )
            return llm.invoke(prompt), model_name
        except Exception as e:
            last_error = e
            # Try next model on 404/removed or any invocation error
            continue
    raise last_error if last_error else RuntimeError("All Cohere model attempts failed.")

def process_user_question(user_question, api_key):
    if not user_question.strip():
        st.warning("Please enter a question.")
        return
    try:
        embeddings = CohereEmbeddings(
            model=EMBEDDING_MODEL,
            cohere_api_key=api_key
        )
        try:
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except FileNotFoundError:
            st.error("Please upload and process PDF files first.")
            return
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return

        docs = new_db.similarity_search(user_question, k=3)
        doc_contents = "\n\n".join([doc.page_content for doc in docs])

        # Allow overriding the model via environment variable after dotenv load
        preferred_model = os.getenv("COHERE_CHAT_MODEL") or CHAT_MODEL
        prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        formatted_prompt = prompt_template.format(context=doc_contents, question=user_question)
        with st.spinner("Generating response..."):
            try:
                response, used_model = _invoke_with_fallback(api_key, preferred_model, formatted_prompt)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.info("Try setting COHERE_CHAT_MODEL to a supported model like 'command-a-03-2025'.")
                return
            if response and getattr(response, "content", "").strip():
                st.write("Reply: ", response.content)
            else:
                st.error("Unable to generate a response. Please try again.")

            
            
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.error(traceback.format_exc())

def main():
    api_key = load_api_key()

    st.markdown("""
        <style>
        .stApp { background-color: #18191A !important; }
        .main, .block-container { background-color: #18191A !important; color: #F5F6FA !important; }
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #4CAF50 0%, #00b894 100%);
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            border: none;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            background-color: #242526;
            color: #F5F6FA;
            font-size: 1.1rem;
            padding: 10px;
        }
        .css-1d391kg { padding: 1.5rem; }
        .sidebar .sidebar-content { background-color: #23272F !important; color: #F5F6FA !important; }
        h1, h2, h3, h4, h5, h6, .stTitle { color: #F5F6FA !important; }
        .header-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: 1px;
            color: #00b894;
            margin-bottom: 0.5rem;
        }
        .header-subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #b2bec3;
            margin-bottom: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='header-title'>Multi-PDF Chatbot 🤖</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-subtitle'>Chat with your PDFs in style. Upload, process, and ask anything!</div>", unsafe_allow_html=True)
    st.write("---")

    with st.container():
        st.markdown("<div class='chat-card'>", unsafe_allow_html=True)
        user_question = st.text_input("Ask a question about your PDFs:", key="user_input")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("img/robots.jpg", width=300)
        st.title("PDF Files")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=['pdf'])
        if st.button("Process PDFs", key="process_button"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
            with st.spinner("Processing PDFs..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the PDF files.")
                        return
                    text_chunks = get_text_chunks(raw_text)
                    if get_vector_store(text_chunks, api_key):
                        st.success("PDFs processed successfully!")
                    else:
                        st.error("Failed to process PDFs.")
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
                    st.error(traceback.format_exc())

    if user_question:
        process_user_question(user_question, api_key)

if __name__ == "__main__":
    main()
