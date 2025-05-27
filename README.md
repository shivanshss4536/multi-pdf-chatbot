# Multi-PDF Chatbot 🤖

A powerful chatbot application that can process multiple PDF files and answer questions about their content using Cohere's AI models.

## Features

- 📚 Process multiple PDF files simultaneously
- 🔍 Extract and analyze text from PDFs
- 💬 Interactive chat interface
- 🎯 Accurate answers based on PDF content
- 🚀 Fast and efficient processing
- 🎨 Beautiful and responsive UI

## Prerequisites

- Python 3.8 or higher
- Cohere API key (Get it from [Cohere's website](https://cohere.com/))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-pdf-chatbot.git
cd multi-pdf-chatbot
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `key.env` file in the project root and add your Cohere API key:
```
COHERE_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Upload your PDF files using the sidebar

4. Click "Submit & Process" to analyze the PDFs

5. Ask questions about the content in the chat interface

## Project Structure

```
multi-pdf-chatbot/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── .gitignore         # Git ignore file
└── img/              # Image assets
```

## How It Works

1. **PDF Processing**: The application extracts text from uploaded PDF files
2. **Text Chunking**: Large texts are split into manageable chunks
3. **Vector Store**: Text chunks are converted into vectors using Cohere's embeddings
4. **Question Answering**: When you ask a question, the app:
   - Finds relevant text chunks
   - Uses Cohere's AI to generate accurate answers
   - Displays the response in a user-friendly format

## Technologies Used

- Streamlit: Web interface
- LangChain: Framework for LLM applications
- Cohere: AI models for embeddings and chat
- FAISS: Vector similarity search
- PyPDF2: PDF text extraction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Cohere for providing the AI models
- Streamlit for the web framework
- LangChain for the LLM framework