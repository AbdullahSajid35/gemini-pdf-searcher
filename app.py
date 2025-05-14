import google.generativeai as genai
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import streamlit as st
import uuid

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY", "8833b2cc-d688-4715-9876-c7c66a361586")
pc = Pinecone(api_key=pinecone_api_key)

# Streamlit session state initialization
if 'mapping' not in st.session_state:
    st.session_state.mapping = None

if 'processed' not in st.session_state:
    st.session_state.processed = False

if 'pdf_loaded' not in st.session_state:
    st.session_state.pdf_loaded = False

if 'index' not in st.session_state:
    st.session_state.index = None

def text_splitter(doc, chunk_size=4000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(doc)

def update_index(document, index):
    with st.spinner("Processing PDF and upserting embeddings..."):
        progress_bar = st.progress(0.0)
        embedding_lists = []
        st.session_state.mapping = {}
        for i, chunk in enumerate(document):
            chunk_content = "Empty" if chunk.page_content == '' else chunk.page_content
            result = genai.embed_content(
                model="models/embedding-001",
                content=chunk_content,
                task_type="retrieval_document",
                title=f"Chunk {i} of PDF"
            )
            chunk_id = f'chunk_{uuid.uuid4()}'  # Unique ID for each chunk
            st.session_state.mapping[chunk_id] = chunk_content
            dictionary = {
                'id': chunk_id,
                'values': result['embedding'],
                'metadata': {'text': chunk_content[:500]}  # Store partial text for metadata
            }
            embedding_lists.append(dictionary)
            progress_bar.progress((i + 1) / len(document))
        
        # Batch upsert embeddings to Pinecone
        try:
            index.upsert(vectors=embedding_lists)
            st.success("Embeddings successfully added to Pinecone index!")
        except Exception as e:
            st.error(f"Failed to upsert embeddings: {str(e)}")
            return None
    return st.session_state.mapping

def get_similar_chunks(query, index, mapping, top_k=5):
    result = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    matches = index.query(vector=result['embedding'], top_k=top_k, include_values=False).matches
    matching_ids = [match['id'] for match in matches]
    matching_chunks = [mapping.get(match_id, "Chunk not found") for match_id in matching_ids]
    text = '\n'.join(matching_chunks)
    return text

def get_response(query, index, stream=True):
    docs = get_similar_chunks(query, index, st.session_state.mapping)
    docs += f'\nOn the basis of the above text, answer: {query}. Please give exact answer as in context not use your own wordings'
    return llm_model.generate_content(docs, stream=stream)

def get_index_name(name):
    return name.name.lower().replace('.pdf', '').replace(' ', '-').replace('_', '-')

def on_uploading():
    st.session_state.pdf_loaded = False
    st.session_state.processed = False

st.markdown('<div style="text-align:center;"><h1 style="font-size:40px;">PDF Searcher <span style="font-size:40px;">🔎</span></h1></div>', unsafe_allow_html=True)
pdf_file = st.file_uploader("Upload PDF file to query questions from it", type=['.pdf'], on_change=on_uploading)

def upload_to_db():
    upload_and_processed()
    if not st.session_state.processed and st.session_state.pdf_loaded:
        with st.spinner("Splitting PDF..."):
            document = PyPDFLoader('temp_pdf_file.pdf').load()
            splitted_document = text_splitter(document)
            index_name = get_index_name(pdf_file)
            
            # Check if index exists, create if it doesn't
            existing_indexes = pc.list_indexes().names()
            if index_name not in existing_indexes:
                if len(existing_indexes) > 0:
                    older_index = existing_indexes[0]
                    try:
                        pc.delete_index(older_index)
                    except Exception as e:
                        st.error(f"Failed to delete older index: {str(e)}")
                        return
                try:
                    pc.create_index(
                        name=index_name,
                        dimension=768,
                        metric='cosine',
                        spec=ServerlessSpec(cloud='aws', region='us-east-1')
                    )
                    st.success(f"Created new Pinecone index: {index_name}")
                except Exception as e:
                    st.error(f"Failed to create index: {str(e)}")
                    return
            
            # Wait for index to be ready
            while index_name not in pc.list_indexes().names():
                pass
            
            st.session_state.index = pc.Index(index_name)
            
            # Update index with embeddings
            st.session_state.mapping = update_index(splitted_document, st.session_state.index)
            if st.session_state.mapping:
                st.session_state.processed = True
            else:
                st.error("Failed to process and index PDF.")

def upload_and_processed():
    if pdf_file is not None and not st.session_state.pdf_loaded:
        with st.spinner("Uploading PDF..."):
            with open("temp_pdf_file.pdf", "wb") as f:
                f.write(pdf_file.getvalue())
        st.session_state.pdf_loaded = True
        pdf_process_button = st.button("Click to process and index PDF", on_click=upload_to_db)

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
            response.resolve()
            st.write(response.text)
            
            # for chunk in response:
            #     st.write(chunk.text)

st.markdown(
    """
    <footer style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px;">
        Developed by Abdullah Sajid
    </footer>
    """,
    unsafe_allow_html=True
)