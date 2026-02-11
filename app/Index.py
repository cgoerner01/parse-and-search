import streamlit as st
from services.indexer_api_client import IndexerAPIClient, IndexDocumentStoreConfig
from pathlib import Path
from utils import save_uploaded_files
import os

# Initialize state
if "converted_documents" not in st.session_state:
    st.session_state.converted_documents = None


st.title("Document Indexing")

st.write("Use your previously converted documents or upload new text files to index them for semantic search.")
indexer = IndexerAPIClient(base_url=os.getenv("INDEXER_API_BASE_URL", "http://localhost:8002"))

uploaded_files = st.file_uploader("Upload Documents", type=["txt"], accept_multiple_files=True)
if st.button("Index Uploaded Documents") and (uploaded_files is not None) and (len(uploaded_files) > 0):
    paths = save_uploaded_files(os.getenv("INDEXED_FILES_PATH", "/data/uploads"), uploaded_files)
    with st.spinner("Indexing documents..."):
        job_status = indexer.index_files(
            file_paths=paths,
            pipeline_type=os.getenv("INDEXER_PIPELINE_TYPE", "simple"),
            document_store_config=IndexDocumentStoreConfig(
                connection_string=os.getenv("PG_CONN_STR", ""),
                embedding_dimension=int(os.getenv("INDEXER_DOCUMENT_STORE_EMBEDDING_DIMENSION", "1024")),
                language=os.getenv("INDEXER_DOCUMENT_STORE_LANGUAGE", "german"),
                vector_function=os.getenv("INDEXER_DOCUMENT_STORE_VECTOR_FUNCTION", "cosine_similarity"),
                recreate_table=os.getenv("INDEXER_DOCUMENT_STORE_RECREATE_TABLE", "False").lower() == "true",
                search_strategy=os.getenv("INDEXER_DOCUMENT_STORE_SEARCH_STRATEGY", "exact_nearest_neighbor")
            ),
            wait=True
        )
        st.write(f"Indexing job status: {job_status.status}")
        st.write(f"Indexing job error (if any): {job_status.error}")
        st.write(f"Indexing job message (if any): {job_status.message}")
    if job_status.status == "failed":
        st.error(f"Indexing failed: {job_status.error}")
        st.stop()
    else:
        st.success("Documents indexed successfully!")

index_converted_documents = st.button("Index Previously Converted Documents", disabled=st.session_state.converted_documents is None)
if st.session_state.converted_documents is not None and index_converted_documents:
    with st.spinner("Indexing documents..."):
        indexer.index_files(
            file_paths=st.session_state.converted_documents,
            pipeline_type=os.getenv("INDEXER_PIPELINE_TYPE", "simple"),
            document_store_config=IndexDocumentStoreConfig(
                connection_string=os.getenv("PG_CONN_STR", ""),
                embedding_dimension=int(os.getenv("INDEXER_DOCUMENT_STORE_EMBEDDING_DIMENSION", "1024")),
                language=os.getenv("INDEXER_DOCUMENT_STORE_LANGUAGE", "german"),
                vector_function=os.getenv("INDEXER_DOCUMENT_STORE_VECTOR_FUNCTION", "cosine_similarity"),
                recreate_table=os.getenv("INDEXER_DOCUMENT_STORE_RECREATE_TABLE", "False").lower() == "true",
                search_strategy=os.getenv("INDEXER_DOCUMENT_STORE_SEARCH_STRATEGY", "exact_nearest_neighbor")
            ),
            wait=True
        )
    if job_status.status == "failed":
        st.error(f"Indexing failed: {job_status.error}")
        st.stop()
    else:
        st.success("Documents indexed successfully!")
