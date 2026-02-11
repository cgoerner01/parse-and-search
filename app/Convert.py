import streamlit as st
from services.converter_api_client import PDFConverterAPIClient
from services.downloading import ZipDownloader
from pathlib import Path
from utils import save_uploaded_files
import os
import io

INDEXED_DOCS_PATH = os.getenv("INDEXED_DOCS_PATH", "./indexed_docs")
UPLOADED_FILES_PATH = os.getenv("UPLOADED_FILES_PATH", "./uploaded_files")
PREPROCESSED_FILES_PATH = os.getenv("PREPROCESSED_FILES_PATH", "./preprocessed_files")
RAPIDOCR_MODELS_PATH = os.getenv("RAPIDOCR_MODELS_PATH", "")

# Initialize state
if "converted_documents" not in st.session_state:
    st.session_state.converted_documents = None

if "show_index_button" not in st.session_state:
    st.session_state.show_index_button = False

st.title("PDF Scan to Text Converter")

st.write("Upload one or more PDF scans to convert them to text files. Then, you can choose to index them to enable semantic search.")

uploaded_files = st.file_uploader("Upload Documents", type=["pdf"], accept_multiple_files=True)

if st.button("Convert to TXT") and (uploaded_files is not None) and (len(uploaded_files) > 0):
    paths = save_uploaded_files(os.getenv("UPLOADED_FILES_PATH", "/data/uploads"), uploaded_files)

    converter = PDFConverterAPIClient(base_url=os.getenv("CONVERTER_API_BASE_URL", "http://localhost:8001"))
    
    zip_downloader = ZipDownloader(input_dir=Path(INDEXED_DOCS_PATH))
    with st.spinner("Converting documents..."):
        #converted_documents = converter.convert_documents(paths)["file_saver"]["documents"]
        job_status = converter.convert_pdfs(
            pdf_paths=paths,
            pipeline_type=os.getenv("CONVERTER_PIPELINE_TYPE", "rapidocr"),
            wait=True
        )
        if job_status.status != "completed":
            st.error(f"Conversion failed: {job_status.error}")
            st.stop()
        converted_documents = [Path(f) for f in job_status.output_files]
        st.session_state.converted_documents = converted_documents
        st.session_state.show_index_button = True
    
    st.success("Documents converted successfully!")
    st.download_button("Download Converted Documents", data=io.BytesIO(open(zip_downloader.zip_input_dir(), "rb").read()), file_name="converted_documents.zip", on_click="ignore")