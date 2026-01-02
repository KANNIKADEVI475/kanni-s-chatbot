 Study companion - A RAG based ChatBot:
 An AI-powered study companion that allows users to upload documents in PDF format and answers the queries based on the retrieved relevant information from documents.
 It reads the document, splits it into chunks, builds embeddings using HuggingFace, and retrieves the most relevant information to answer user queries.
 
 Tech Stack:
 1.Python
 2.Streamlit
 3.PyPDF2
 4.LangChain
 5.HuggingFace Sentence Transformers
 6.FAISS Vector Database
 7.gemini-2.5-flash

  How It Works - RAG Pipeline:
 1️. Extract text from uploaded PDF
 2️. Split text into chunks
 3️. Convert chunks into embeddings
 4️. Store in FAISS vector store
 5️. When a user asks a question
 6️. Retrieve most relevant chunks
 7️. Pass them to LLM to generate answer
