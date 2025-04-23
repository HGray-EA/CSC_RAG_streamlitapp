import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint 
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
import os
import base64
import tempfile  
import requests

st.set_page_config(page_title="Supporting Citizen Science", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>EA Supporting Citizen Science TAF</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Got a question about the EA's Supporting Citizen Science technical advisory framework? <br>" \
    "Use this app to question it below!",
    unsafe_allow_html=True
)
st.markdown(" ")
# Hugging Face API Key
HUGGINGFACEHUB_API_TOKEN = st.secrets["huggingface"]["token"]

# ---------------------- Load and Split PDF ---------------------

def process_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_pdf_path = tmp_file.name
    
    # Use PyMuPDFLoader correctly by loading the temporary PDF file
    loader = PyMuPDFLoader(tmp_pdf_path)
    pages = loader.load()  # This should return the pages as documents
    if not pages:
        raise ValueError("No pages were loaded from the PDF.")

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    return chunks


# -----------------------  Build Vector Store ----------------------

def build_vector_store(chunks):
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",  
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
#def build_vector_store(chunks):
 #   embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  #  vectorstore = FAISS.from_documents(chunks, embeddings)
   # return vectorstore

# -------------------------- Customise agent response ----------------------
#  We make sure to cite rules
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""

Please answer the question based on the provided context in clear, concise, and readable text. Quote chapters and page numbers you find your information on.
Use the provided context to answer the question

#Context:
#{context}

#Question:
#{question}

AI Agent response:
#"""
)


# Can add a banner 

# Banner
#image_url = "https://avatars.githubusercontent.com/u/123268593?v=4"  # URL of your image or local path
#st.markdown(f"""
#    <div style="text-align: center;">
#        <img src="{image_url}" width="100%" alt="Banner Image">
#    </div>
#""", unsafe_allow_html=True)



# Load PDF from GitHub
pdf_url = "https://raw.githubusercontent.com/HGray-EA/CSC_RAG_streamlitapp/main/Technical_Advisory_Framework_CSC_EA.pdf"
response = requests.get(pdf_url)

if response.status_code == 200:
    pdf_bytes = response.content

# -------------------------------- Display PDF --------------------

    # Then wrap the viewer in the div with the class pdf-viewer
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_viewer_file:
        tmp_viewer_file.write(pdf_bytes)
        tmp_pdf_path = tmp_viewer_file.name
        
        # Center the pdf_viewer using Streamlit columns & display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pdf_viewer(tmp_pdf_path, height=600, width=900)

    
    # Process and index the PDF
    with st.spinner("Processing PDF..."):
        chunks = process_pdf(pdf_bytes)
        vectorstore = build_vector_store(chunks)
        retriever = vectorstore.as_retriever()

        # Setup the model
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation", 
            temperature=0.2,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )

        st.success("PDF loaded and indexed. You can now ask questions!")

        # Chat UI
        st.subheader("Question the document")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Show chat history
        for speaker, msg in st.session_state.chat_history:
            if speaker == "You":
                with st.chat_message("user"):
                    st.markdown(msg)
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**AI Agent**\n\n{msg}")

        # User input
        user_input = st.chat_input("Ask a question about the framework...")

        if user_input:
            # Display user's message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Process with QA chain
            with st.spinner("Getting response..."):
                result = qa_chain({"query": user_input})
                answer = result["result"]

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(f"**AI Agent:**\n\n{answer}")

            # Save to chat history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI Agent", answer))
    
