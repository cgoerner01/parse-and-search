import streamlit as st
import streamlit_scrollable_textbox as stx
from services.converter_api_client import PDFConverterAPIClient
from services.downloading import ZipDownloader
from pathlib import Path
from utils import save_uploaded_files
import os
import io
import fitz
import pandas as pd
from typing import List
from datetime import datetime

INDEXED_DOCS_PATH = os.getenv("INDEXED_DOCS_PATH", Path(os.getcwd()).resolve().parent / "converter" / "app" / "outputs")
UPLOADED_FILES_PATH = os.getenv("UPLOADED_FILES_PATH", Path(os.getcwd()).resolve().parent / "converter" / "app" / "uploads")
PREPROCESSED_FILES_PATH = os.getenv("PREPROCESSED_FILES_PATH", Path(os.getcwd()).resolve().parent / "converter" / "app" / "preprocessed")
RAPIDOCR_MODELS_PATH = os.getenv("RAPIDOCR_MODELS_PATH", "")
OUTPUT_JSON_PATH = os.getenv("OUTPUT_JSON_PATH", Path(os.getcwd()).resolve().parent / "converter" / "app" / "outputs" / "output.json")

# Initialize state
if "converted_documents" not in st.session_state:
    st.session_state.converted_documents = None

if "show_index_button" not in st.session_state:
    st.session_state.show_index_button = False

if "job_status" not in st.session_state:
    st.session_state.job_status = None

if "not_yet_processed_files" not in st.session_state:
    st.session_state.not_yet_processed_files = None

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

def add_testcases_to_output_json(output_json_path: Path, paths: List[Path]):
    output_df = pd.read_json(output_json_path, orient="index")
    if output_df.empty:
        output_df = pd.DataFrame(columns=["patient_id", "document_id"])
    for path in paths:
        document_id = Path(path).stem
        if document_id in output_df["document_id"].values:
            continue
        patient_id = document_id.split("_")[0]
        row_to_append = pd.DataFrame({"patient_id": [patient_id], "document_id": [document_id]})
        output_df = pd.concat([output_df, row_to_append], ignore_index=True)
    output_df[str(ocr_backend) + "_text"] = output_df.get(str(ocr_backend) + "_text", None)
    output_df[str(ocr_backend) + "_execution_time"] = output_df.get(str(ocr_backend) + "_execution_time", None)
    output_df[str(ocr_backend) + "_comments"] = output_df.get(str(ocr_backend) + "_comments", None)
    output_df.to_json(output_json_path, orient="index")

def get_not_yet_processed_files(uploaded_files: List[Path], output_json_path: Path) -> List[Path]:
    # either document_id does not exist or it exists but the text column for the selected OCR backend is null
    output_df = pd.read_json(output_json_path, orient="index")
    processed_document_ids = set(output_df[~output_df[ocr_backend + "_text"].isnull()]["document_id"].tolist())
    not_yet_processed_files = [Path(path) for path in uploaded_files if Path(path).stem not in processed_document_ids]
    st.session_state.not_yet_processed_files = not_yet_processed_files
    return not_yet_processed_files

def save_converted_documents_to_json(output_json_path: Path, converted_documents: List[Path]):
    output_df = pd.read_json(output_json_path, orient="index")
    for converted_document in converted_documents:
        document_id = Path(converted_document).stem
        patient_id = document_id.split("_")[0]
        output_df.loc[output_df["document_id"] == document_id, ocr_backend + "_text"] = open(Path(converted_document), 'r').read()
        output_df.loc[output_df["document_id"] == document_id, ocr_backend + "_execution_time"] = datetime.fromisoformat(st.session_state.job_status.completed_at) - datetime.fromisoformat(st.session_state.job_status.created_at)
        output_df.loc[output_df["document_id"] == document_id, ocr_backend + "_comments"] = ""
    output_df.to_json(output_json_path, orient="index")


if uploaded_files is not None and len(uploaded_files) > 0:
    paths = save_uploaded_files(UPLOADED_FILES_PATH, uploaded_files)
    add_testcases_to_output_json(OUTPUT_JSON_PATH, paths)


run_single_conversion_button = st.button("Run single conversion", disabled=(uploaded_files is None or len(uploaded_files) == 0))
run_all_conversion_button = st.button("Run conversion for all uploaded files", disabled=(uploaded_files is None or len(uploaded_files) == 0))

select_col1, select_col2 = st.columns([7,3])

with select_col1:
    if len(uploaded_files) > 0:
        get_not_yet_processed_files(paths, OUTPUT_JSON_PATH)
        selected_uploaded_file = st.selectbox(label="Uploaded files for preview:", options=st.session_state.not_yet_processed_files)
with select_col2:
    if len(uploaded_files) > 0:
        max_value = get_pdf_page_count([path for path in paths if path == selected_uploaded_file][0]) if selected_uploaded_file else 1
        preview_page_number = st.number_input("Page:", min_value=1, max_value=max_value, step=1, key="preview_page_number", disabled=(selected_uploaded_file is None))
        st.write(f"Page count: {max_value}")

preview_col1, preview_col2, preview_col3 = st.columns(3)

with preview_col1:
    if len(uploaded_files) > 0:
        if selected_uploaded_file:
            st.image(get_pdf_page_as_img([path for path in paths if path == selected_uploaded_file][0], preview_page_number), caption="Preview of selected page")
    if (uploaded_files is not None) and (len(uploaded_files) > 0) and selected_uploaded_file is not None:
        converter = PDFConverterAPIClient(base_url=os.getenv("CONVERTER_API_BASE_URL", "http://localhost:8001"))
        
        if run_single_conversion_button:
            with st.spinner("Converting documents..."):
                
                st.session_state.job_status = converter.convert_pdfs(
                    pdf_paths=[Path(selected_uploaded_file)],
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
        if run_all_conversion_button:
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
                save_converted_documents_to_json(OUTPUT_JSON_PATH, st.session_state.converted_documents)
                get_not_yet_processed_files(paths, OUTPUT_JSON_PATH)
    if st.session_state.converted_documents is not None:
        st.success("Documents converted successfully!")
        zip_downloader = ZipDownloader(input_dir=Path(INDEXED_DOCS_PATH) / st.session_state.job_status.job_id)
        st.download_button("Download Converted Documents", data=io.BytesIO(open(zip_downloader.zip_input_dir(), "rb").read()), file_name="converted_documents.zip", on_click="ignore")


with preview_col2:
    if st.session_state.converted_documents != None:
        selected_converted_document_path = [file for file in st.session_state.converted_documents if file.stem == Path(selected_uploaded_file).stem][0]
        stx.scrollableTextbox(open(selected_converted_document_path, 'r').read(), height=400)

with preview_col3:
    if st.session_state.converted_documents != None:
        output_df = pd.read_json(OUTPUT_JSON_PATH, orient="index")
        comments = st.text_area("Comments", height=400)
        if st.button("Save to JSON"):
            save_converted_documents_to_json(OUTPUT_JSON_PATH, st.session_state.converted_documents)
            st.session_state.converted_documents = None
            get_not_yet_processed_files(paths, OUTPUT_JSON_PATH)


