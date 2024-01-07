import google.generativeai as genai
import os
import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm_model=genai.GenerativeModel('gemini-pro')
pinecone.init(api_key='8833b2cc-d688-4715-9876-c7c66a361586',environment='gcp-starter')
spinner = st.spinner('PDF Processing...')
if 'mapping' not in st.session_state:
    st.session_state.mapping=None

if 'processed' not in st.session_state:
    st.session_state.processed=False

if 'pdf_loaded' not in st.session_state:
    st.session_state.pdf_loaded=False

if 'index' not in st.session_state:
    st.session_state.index=None



def text_splitter(doc,chunk_size=4000,chunk_overlap=100):
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_documents(doc)

def update_index(document,index):
    with st.spinner("PDF Processing....."):
        progress_bar=st.progress(0.0)
        embedding_lists=[]
        st.session_state.mapping={}
        for i,chunk in enumerate(document):
            chunk="Empyt" if chunk.page_content=='' else chunk.page_content
            result = genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="retrieval_document",
            title="Embedding of single string")
            st.session_state.mapping[f'chunk_{i}']=chunk
            dictionary={'id':f'chunk_{i}','values':result['embedding']}
            embedding_lists.append(dictionary)
            progress_bar.progress((i+1)/len(document))
        index.upsert(vectors=embedding_lists)
    return st.session_state.mapping



def get_similar_chunks(query,index,mapping,top_k=5):
    result = genai.embed_content(
    model="models/embedding-001",
    content=query,
    task_type="retrieval_query")
    matches= index.query(result['embedding'],top_k=top_k).matches
    matching_ids=[match['id'] for match in matches]
    matching_chunks= [mapping[match_id] for match_id in  matching_ids]
    text=''
    for chunk in matching_chunks:
        text+=chunk
    return text


def get_response(query,index,stream=True):
    docs=get_similar_chunks(query,index,st.session_state.mapping)
    st.write()
    docs+=f'   \nOn the base of above whole text give me answer that {query}? '
    return llm_model.generate_content(docs,stream=stream)

def get_index_name(name):
    return name.name.lower().replace('.pdf','').replace(' ','-').replace('_','-')

def on_uploading():
    st.session_state.pdf_loaded=False
    st.session_state.processed=False
    
st.markdown('<div style="text-align:center;"><h1 style="font-size:40px;">PDF Searcher <span style="font-size:40px;">🔎</span></h1></div>', unsafe_allow_html=True)
pdf_file=st.file_uploader("Upload PDF file to query questions from it",type=['.pdf'],on_change=on_uploading)



def upload_to_db():
   upload_and_processed() 
   if not st.session_state.processed:
        with st.spinner("PDF Splitting......"):
            document=PyPDFLoader('temp_pdf_file.pdf').load()
            splitted_document=text_splitter(document)
            index_name=get_index_name(pdf_file)
            st.write(index_name)
            st.session_state.index=pinecone.Index(index_name)
            if index_name not in pinecone.list_indexes():
                if len(pinecone.list_indexes())>0:
                    older_index=pinecone.list_indexes()[0]
                    pinecone.delete_index(older_index)
                pinecone.create_index(index_name,768)
            else:
                st.warning("File is already processed")
                st.session_state.processed=True
                return
        st.session_state.mapping=update_index(splitted_document,st.session_state.index)
        st.session_state.processed=True

def upload_and_processed():
    if pdf_file is not None and not st.session_state.pdf_loaded:
        with st.spinner("PDF Uploading......"):
            with open("temp_pdf_file.pdf", "wb") as f:
                f.write(pdf_file.getvalue())
        st.session_state.pdf_loaded=True
        pdf_process_button=st.button("Click to upload",on_click=upload_to_db)

upload_and_processed()    

if st.session_state.processed:
    question = st.text_area("Ask something from your PDF", placeholder='Enter your question here')
    submit = st.button("Get your answer")

    if submit:
        if not question:
            st.error("Please enter a question")
        elif not pdf_file:
            st.error("Please upload PDF")

        else:
            response = get_response(question, st.session_state.index)
            for chunk in response:
                st.write(chunk.text)


st.markdown(
    """
    <footer style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px;">
        Developed by Abdullah Sajid
    </footer>
    """,
    unsafe_allow_html=True
)
    


