import streamlit as st
import streamlit_scrollable_textbox as stx
from services.converter_api_client import PDFConverterAPIClient
from services.downloading import ZipDownloader
from pathlib import Path
from utils import save_uploaded_files
import os
import io
import fitz

INDEXED_DOCS_PATH = os.getenv("INDEXED_DOCS_PATH", Path(os.getcwd()).resolve().parent / "converter" / "app" / "outputs")
UPLOADED_FILES_PATH = os.getenv("UPLOADED_FILES_PATH", Path(os.getcwd()).resolve().parent / "converter" / "app" / "uploads")
PREPROCESSED_FILES_PATH = os.getenv("PREPROCESSED_FILES_PATH", Path(os.getcwd()).resolve().parent / "converter" / "app" / "preprocessed")
RAPIDOCR_MODELS_PATH = os.getenv("RAPIDOCR_MODELS_PATH", "")

# Initialize state
if "converted_documents" not in st.session_state:
    st.session_state.converted_documents = None

if "show_index_button" not in st.session_state:
    st.session_state.show_index_button = False

if "job_status" not in st.session_state:
    st.session_state.job_status = None

def get_pdf_page_count(pdf_file_path):
    """Utility function to get the number of pages in a PDF file."""
    with fitz.open(pdf_file_path) as doc:
        print(doc.page_count)
        return doc.page_count

def get_pdf_page_as_img(pdf_file_path, page_number):
    """Utility function to get a specific page of a PDF as an image."""
    with fitz.open(pdf_file_path) as doc:
        if 1 <= page_number <= doc.page_count:
            page = doc.load_page(page_number - 1)  # Convert to 0-based indexing
            pix = page.get_pixmap(dpi=300)
            return pix.tobytes("ppm")
        else:
            raise ValueError(f"Page number {page_number} is out of range for this document.")

margins_css = """
    <style>
        .main > div {
            padding-left: 0rem;
            padding-right: 0rem;
        }
    </style>
"""

st.markdown(margins_css, unsafe_allow_html=True)

st.title("PDF Scan to Text Converter")

st.write("Upload one or more PDF scans to convert them to text files. Then, you can choose to index them to enable semantic search.")

ocr_backend = st.selectbox("Select OCR Backend", options=["rapidocr", "suryaocr", "tesseract", "easyocr", "docling_easyocr", "vlm", "macocr", "glm-ocr", "deepseek-ocr"], index=0, help="Choose the OCR backend to use for text extraction. Currently, only RapidOCR is supported.")

uploaded_files = st.file_uploader("Upload Documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) > 0:
    paths = save_uploaded_files(UPLOADED_FILES_PATH, uploaded_files)

run_text_conversion_button = st.button("Run text conversion", disabled=(uploaded_files is None or len(uploaded_files) == 0))

select_col1, select_col2 = st.columns([7,3])

with select_col1:
    selected_uploaded_file = st.selectbox(label="Uploaded files for preview:", options=[file.name for file in uploaded_files])
with select_col2:
    max_value = get_pdf_page_count([path for path in paths if path.name == selected_uploaded_file][0]) if selected_uploaded_file else 1
    preview_page_number = st.number_input("Page:", min_value=1, max_value=max_value, step=1, key="preview_page_number", disabled=(selected_uploaded_file is None))

preview_col1, preview_col2, preview_col3 = st.columns(3)

with preview_col1:
    if selected_uploaded_file:
        st.image(get_pdf_page_as_img([path for path in paths if path.name == selected_uploaded_file][0], preview_page_number), caption="Preview of selected page")
    if run_text_conversion_button and (uploaded_files is not None) and (len(uploaded_files) > 0):
        converter = PDFConverterAPIClient(base_url=os.getenv("CONVERTER_API_BASE_URL", "http://localhost:8001"))
        
        with st.spinner("Converting documents..."):
            
            st.session_state.job_status = converter.convert_pdfs(
                pdf_paths=paths,
                #pipeline_type=os.getenv("CONVERTER_PIPELINE_TYPE", "rapidocr"),
                pipeline_type=ocr_backend,
                wait=True
            )
            if st.session_state.job_status.status != "completed":
                st.error(f"Conversion failed: {st.session_state.job_status.error}")
                st.stop()
            st.write(f"Job ID: {st.session_state.job_status.job_id}")
            converted_documents = [(Path(INDEXED_DOCS_PATH) / st.session_state.job_status.job_id / f) for f in st.session_state.job_status.output_files]
            print(converted_documents)
            st.session_state.converted_documents = converted_documents
    if st.session_state.converted_documents is not None:
        st.success("Documents converted successfully!")
        zip_downloader = ZipDownloader(input_dir=Path(INDEXED_DOCS_PATH) / st.session_state.job_status.job_id)
        st.download_button("Download Converted Documents", data=io.BytesIO(open(zip_downloader.zip_input_dir(), "rb").read()), file_name="converted_documents.zip", on_click="ignore")

with preview_col2:
    if st.session_state.converted_documents != None:
        try:
            st.image(Path(PREPROCESSED_FILES_PATH) / st.session_state.job_status.job_id / f"vis_{Path(selected_uploaded_file).stem}_page{preview_page_number}.jpg", caption="Visualization of text recognition results")
        except Exception as e:
            st.text("Visualization not available.")

with preview_col3:
    if st.session_state.converted_documents != None:
        selected_converted_document_path = [file for file in st.session_state.converted_documents if file.stem == Path(selected_uploaded_file).stem][0]
        stx.scrollableTextbox(open(selected_converted_document_path, 'r').read(), height=400)
