import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint 
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
import os
import base64
import tempfile  

st.set_page_config(page_title="PDF querier", layout="wide")

# Hugging Face API Key
HUGGINGFACEHUB_API_TOKEN = st.secrets["huggingface"]["token"]

# ---------------------- Load and Split PDF ---------------------
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    return chunks

# -----------------------  Build Vector Store ----------------------
def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# -------------------------- Customise agent response ----------------------
#  We make sure to cite rules
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Special Agent Dale Cooper from Twin Peaks.
You speak with clarity, warmth, and a philosophical bent — often noticing the hidden beauty in things.
You’re calm, deliberate, and sometimes speak as if narrating a dream. 
But when it comes to rules, you're precise and to-the-point; always quoting rule numbers.

Use the provided context to answer the question.

#Context:
#{context}

#Question:
#{question}

Agent Cooper's Response:
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


# Load PDF path
pdf_path = "/home/khadas/Downloads/EHBA ruleset 230712.pdf"  


# -------------------------------- Display PDF --------------------
with open(pdf_path, "rb") as pdf_file:
    pdf_bytes = pdf_file.read()
    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

    pdf_display = f'<div style="display: flex; justify-content: center;"><iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="600"></iframe></div>'
    st.markdown(pdf_display, unsafe_allow_html=True)

    # Process and index the PDF
    with st.spinner("Processing PDF..."):
        chunks = process_pdf(pdf_path)
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

        user_input = st.text_input("Your question:", key="user_question")

        if user_input:
            with st.spinner("Getting answer..."):
                result = qa_chain({"query": user_input})
                answer = result["result"]
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bike Polo Guru", answer))

            # WhatsApp-style chat layout
            for speaker, msg in st.session_state.chat_history[::-1]:
                if speaker == "You":
                    st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                            <div style="max-width: 60%; background-color: #8f9296; border-radius: 10px; padding: 10px; margin-right: 5px; text-align: right;">
                                <strong>{speaker}:</strong> {msg}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div style="max-width: 60%; background-color: #9f4129; border-radius: 10px; padding: 10px; margin-left: 5px;">
                                <strong>{speaker}:</strong> {msg}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
