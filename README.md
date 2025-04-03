# 📚 PDF Chatbot

A powerful interactive chat interface for querying PDF documents using LLM and vector embeddings.

## 🌟 Features

- **📄 PDF Processing**: Load single PDFs or batch process entire directories
- **🔍 Intelligent Retrieval**: Ask questions about your documents and get relevant answers
- **🔗 Context-Aware Responses**: Powered by retrieval-augmented generation (RAG)
- **🧠 Local LLM Integration**: Uses Mistral-7B by default with HuggingFace Hub
- **💾 Persistent Vector Storage**: Saves document embeddings for future sessions
- **🔄 Expandable Document Base**: Add more documents anytime to expand knowledge base

## 🛠️ Technologies

- **LangChain**: Framework for building LLM applications
- **HuggingFace Hub**: Access to state-of-the-art language models
- **ChromaDB**: Vector database for document storage and retrieval
- **PyPDF**: PDF parsing and extraction
- **Sentence Transformers**: Document embedding generation

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- HuggingFace API token (set as environment variable or enter during runtime)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Setting Up HuggingFace API Token

```bash
# Set as environment variable
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

## 📋 Usage

Run the application:

```bash
python pdf_chatbot.py
```

### Main Menu Options

1. **Process PDF File or Directory**: Load your documents into the system
2. **Ask Questions**: Interactive mode to query your documents
3. **Document Statistics**: View information about loaded documents
4. **Clear Documents**: Remove all documents from the system
5. **Exit**: Close the application

## ⚙️ Configuration

You can customize the following parameters in the `PDFChatbot` class:

- `model_name`: LLM model to use (default: "mistralai/Mistral-7B-Instruct-v0.2")
- `embedding_model_name`: Embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")
- `persist_dir`: Directory for storing vector database (default: "./chroma_db")

## 🏗️ Project Structure

```
pdf-chatbot/
├── pdf_chatbot.py        # Main application code
├── chroma_db/            # Vector database storage (created on first run)
├── pdf_chatbot.log       # Application logs
├── requirements.txt      # Dependencies
└── README.md             # This documentation
```

## 📝 Example

```python
# Programmatic usage example
from pdf_chatbot import PDFChatbot

# Initialize the chatbot
chatbot = PDFChatbot()

# Process a PDF
chatbot.process_pdf("path/to/your/document.pdf")

# Ask a question
response = chatbot.ask_question("What are the key findings in this document?")
print(response["result"])
```

## 📊 Performance Considerations

- For best performance, use smaller, focused PDFs
- Processing very large documents may require more memory
- Consider adjusting `chunk_size` and `chunk_overlap` parameters for different document types

## 🔒 Privacy Notice

All document processing happens locally on your machine. No document content is sent to external servers except:
- API calls to HuggingFace for model inference
- Document embeddings are stored locally in the ChromaDB database

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- HuggingFace for providing access to state-of-the-art language models
- LangChain for the comprehensive LLM application framework
- All open-source contributors who made this project possible

---
