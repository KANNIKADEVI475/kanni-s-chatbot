import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = "***********************************"
st.header("Study companion")

with st.sidebar:
    st.title("My files")
    file=st.file_uploader("Upload a file",type="pdf")

if file is not None:
    my_pdf=PdfReader(file)
    text=""
    for page in my_pdf.pages:
        text+=page.extract_text()
        #st.write(text)

    text_spliter=RecursiveCharacterTextSplitter(separators=["\n"],chunk_size=800,chunk_overlap=100)
    chucks=text_spliter.split_text(text)
    #st.write(chucks)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store=FAISS.from_texts(chucks,embeddings)

    user_query=st.text_input("Ask your doubts")

    if user_query:
        matching_chucks=vector_store.similarity_search(user_query)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY
        )

        chain=load_qa_chain(llm,chain_type="stuff")
        output=chain.run(question=user_query,input_documents=matching_chucks)
        st.subheader("Answer:")
        st.write(output)



