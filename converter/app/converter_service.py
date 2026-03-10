"""
Docling Converting Service - extracted from original code for use with FastAPI.
"""

import os
from pathlib import Path
from docling_haystack.converter import DoclingConverter, ExportType
from haystack import Pipeline, Document
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack import component

from docling.chunking import HybridChunker

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions,
    EasyOcrOptions,
    VlmPipelineOptions,
    RapidOcrOptions,
    OcrMacOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption, ConversionResult
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from docling.pipeline.vlm_pipeline import VlmPipeline

from docling_surya import SuryaOcrOptions

from collections import defaultdict

from docling.datamodel import vlm_model_specs

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

from typing import Iterable, Union, List

from modelscope import snapshot_download

from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

from surya.models import load_predictors
from surya.common.surya.schema import TaskNames

import fitz

import logging

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.DEBUG)

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-large-instruct")
EXPORT_TYPE = ExportType.DOC_CHUNKS


@component
class TextFileSaver:
    """Component to save document chunks to text files."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        grouped = defaultdict(list)

        for doc in documents:
            try:
                # TODO: this doesn't work for the VLM pipeline
                source = doc.meta["dl_meta"]["meta"]["origin"]["filename"]
            except KeyError:
                try:
                    source = doc.meta["source"]
                except KeyError:
                    source = "unknown"
            grouped[source].append(doc.content)

        for source_file, contents in grouped.items():
            name = Path(source_file).stem
            out_path = self.output_dir / f"{name}.txt"

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(contents))

        return {"documents": documents}


@component
class Deskewer:
    """Component to deskew PDF pages before OCR processing."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def deskew(self, img):
        """Deskew a single image."""
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Deskew
        coords = np.column_stack(np.where(gray < 200))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            print(f"Detected skew angle: {angle}")
            
            # Rotate
            (h, w) = gray.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                img_cv, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            
            # Add margin to prevent boundary issues
            margin = 50
            bordered = cv2.copyMakeBorder(
                rotated, margin, margin, margin, margin,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            return Image.fromarray(cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB))
        
        # Return original if no coords found
        return img

    def getSkewAngle(self, img) -> float:
        """
        Calculate skew angle of an image.
        Reference: https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
        """
        # Prep image, copy, convert to gray scale, blur, and threshold
        newImage = np.array(img)
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=5)

        # Find all contours
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find largest contour and surround in min area box
        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)

        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    def rotateImage(self, img, angle: float):
        """Rotate the image around its center."""
        newImage = np.array(img)
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(newImage)

    def deskew_old(self, img):
        """Deskew image using alternative method."""
        angle = self.getSkewAngle(img)
        return self.rotateImage(img, -1.0 * angle)
    
    @component.output_types(paths=Iterable[Union[Path, str]])
    def run(self, paths: List[Path]):
        """Process all PDF pages through deskewing."""
        output_paths = []
        
        # Determine poppler path based on system
        poppler_path = os.getenv("POPPLER_PATH", None)
        if poppler_path is None:
            # Try to detect common installation paths
            if os.path.exists("/opt/homebrew/bin"):
                poppler_path = "/opt/homebrew/bin"
            elif os.path.exists("/usr/bin"):
                poppler_path = "/usr/bin"
        
        for path in paths:
            processed_pages = []
            
            # Convert PDF to images
            if poppler_path:
                pages = convert_from_path(path, 400, poppler_path=poppler_path)
            else:
                pages = convert_from_path(path, 400)
            
            # Deskew each page
            for page in pages:
                processed_pages.append(self.deskew(page))
            
            # Save deskewed PDF
            output_path = self.output_dir / Path(path).name
            if processed_pages:
                processed_pages[0].save(
                    output_path, 
                    save_all=True, 
                    append_images=processed_pages[1:] if len(processed_pages) > 1 else [],
                    resolution=400.0
                )
                output_paths.append(output_path)
        
        return {"paths": output_paths}

@component
class RapidOCRConverter:
    """Component to convert PDFs using RapidOCR."""

    def __init__(self, preprocess_dir: str = None):
        self.preprocess_dir = preprocess_dir
        self.engine = RapidOCR(
            params={
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.MULTI,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV4,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.LATIN,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
        )
    
    @component.output_types(documents=list[Document])
    def run(self, paths: List[Path]):
        results = []
        for path in paths:
            pages = []
            with fitz.open(path) as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    res = self.engine(img)
                    visualization = Image.fromarray(res.vis(Path(self.preprocess_dir) / f"vis_{path.stem}_page{page.number+1}.jpg"))
                    w, h = visualization.size
                    visualization = visualization.crop((0, 0, w//2, h))  # Crop to left half to remove bounding boxes
                    visualization.save(Path(self.preprocess_dir) / f"vis_{path.stem}_page{page.number+1}.jpg")
                    pages.append(res)
                results.append(pages)
        
        output_docs = []
        for res, path in zip(results, paths):
            output_docs.append(
                Document(
                    content="\n".join([page.to_markdown() for page in res if hasattr(page, 'to_markdown')]),
                    meta={"source": str(Path(path).name)},
                )
            )
        
        return {"documents": output_docs}

@component
class SuryaOCRConverter:
    """Component to convert PDFs using SuryaOCR."""
    
    def __init__(self, preprocess_dir: str = None):
        self.preprocess_dir = preprocess_dir
        self.predictors = load_predictors()
    
    @component.output_types(documents=list[Document])
    def run(self, paths: List[Path]):
        results = []
        for path in paths:
            pages = []
            with fitz.open(path) as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    res = self.predictors["recognition"](
                        [img],
                        task_names=[TaskNames.ocr_with_boxes],
                        bboxes=None,
                        det_predictor=self.predictors["detection"],
                        highres_images=None,
                        math_mode=True,
                        return_words=True
                    )[0]
                    #visualization = Image.fromarray(res[0].vis(Path(self.preprocess_dir) / f"vis_{path.stem}_page{page.number+1}.jpg"))
                    #w, h = visualization.size
                    #visualization = visualization.crop((0, 0, w//2, h))  # Crop to left half to remove bounding boxes
                    #visualization.save(Path(self.preprocess_dir) / f"vis_{path.stem}_page{page.number+1}.jpg")
                    pages.append(res)
                results.append(pages)

        output_docs = []
        for res, path in zip(results, paths):
            output_docs.append(
                Document(
                    content = '\n'.join(['\n'.join([line.text for line in page.text_lines]) for page in res]),
                    meta={"source": str(Path(path).name)},
                )
            )
        
        return {"documents": output_docs}


class DoclingConvertingService:
    """
    Service for converting PDF documents using Docling.
    Supports multiple pipeline types: OCR, OCR with deskewing, and VLM.
    """
    
    def __init__(self, preprocess_dir: Path = None, output_dir: Path = None, rapidocr_models_path: Path = None):
        self.preprocess_dir = preprocess_dir
        self.output_dir = output_dir
        self.rapidocr_models_path = rapidocr_models_path

        self.convert_pipe = Pipeline()
        self.pipeline_options = None
        self.doc_converter = None
    
    def download_rapidocr_models(self):
        os.makedirs(self.rapidocr_models_path, exist_ok=True)

        if not os.listdir(self.rapidocr_models_path):
            print("Downloading RapidOCR models...")
            snapshot_download(
                repo_id="RapidAI/RapidOCR",
                cache_dir=self.rapidocr_models_path
            )
        else:
            print("Models already present, skipping download.")

    def init_rapidocr_pipeline(self):
        """Initialize OCR pipeline with RapidOCR."""
        """
        self.download_rapidocr_models()

        det_model_path = os.path.join(
            self.rapidocr_models_path, "RapidAI", "RapidOCR", "onnx", "PP-OCRv4", "det", "en_PP-OCRv3_det_infer.onnx"
        )
        rec_model_path = os.path.join(
            self.rapidocr_models_path, "RapidAI", "RapidOCR", "onnx", "PP-OCRv5", "rec", "latin_PP-OCRv5_rec_mobile_infer.onnx"
        )
        cls_model_path = os.path.join(
            self.rapidocr_models_path, "RapidAI", "RapidOCR", "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        )
        
        ocr_options = RapidOcrOptions(
            force_full_page_ocr=True,
            bitmap_area_threshold=0.0,  # Adjust as needed
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            cls_model_path=cls_model_path,
        )
        
        self.pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr_options,
        )

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                ),
            },
        )

        self.convert_pipe.add_component(
            "converter",
            DoclingConverter(
                converter=self.doc_converter,
                export_type=EXPORT_TYPE,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=512),
            ),
        )
        """

        self.convert_pipe.add_component(
            "converter",
            RapidOCRConverter(preprocess_dir=self.preprocess_dir),
        )

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        #self.convert_pipe.add_component(
        #    "cleaner",
        #    DocumentCleaner(),
        #)
        #   
        #self.convert_pipe.add_component(
        #    "splitter",
        #    DocumentSplitter(split_by="word", split_length=384, split_overlap=10),
        #)

        self.convert_pipe.connect("converter", "file_saver")
        #self.convert_pipe.connect("file_saver", "cleaner")
        #self.convert_pipe.connect("cleaner", "splitter")
    
    def init_surya_pipeline(self):
        """Initialize OCR pipeline with SuryaOCR."""
        """
        surya_options = SuryaOcrOptions(
            lang=["de"],
        )
        self.pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_model="suryaocr",
            allow_external_plugins=True,
            ocr_options=surya_options,
            bitmap_area_threshold=0.0,  # Adjust as needed
        )

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options),
            }
        )

        self.convert_pipe.add_component(
            "converter",
            DoclingConverter(
                converter=self.doc_converter,
                export_type=EXPORT_TYPE,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=512),
            ),
        )

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )
        """
        self.convert_pipe.add_component(
            "converter",
            SuryaOCRConverter(preprocess_dir=self.preprocess_dir),
        )

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )
        self.convert_pipe.connect("converter", "file_saver")
    
    def init_macocr_pipeline(self):
        """Initialize OCR pipeline with MacOCR."""
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options = OcrMacOptions(
            force_full_page_ocr=True,
            lang=["de-DE"],
        )
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                    backend=PyPdfiumDocumentBackend
                ),
            },
        )

        self.convert_pipe.add_component(
            "converter",
            DoclingConverter(
                converter=self.doc_converter,
                export_type=EXPORT_TYPE,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=512),
            ),
        )

        self.convert_pipe.add_component(
            "cleaner",
            DocumentCleaner(
                remove_substrings=[
                    "999999 Elektronisches Dokument",
                    "Keine Archivierung des Ausdrucks am UKHD"
                ],
                replace_regexes={
                    r'Ausdruck aus.* Unterbelegart:.*(riefe|richte)' : '',
                    r'Gedruckte Seiten.*Datum: \d{2}\.\d{2}\.\d{4}' : '',
                }
            ),
        )

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("converter", "cleaner")
        self.convert_pipe.connect("cleaner", "file_saver")
    
    def init_docling_easyocr_pipeline(self):
        """Initialize OCR pipeline with Docling's EasyOCR integration."""
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options = EasyOcrOptions(
            force_full_page_ocr=True,
            lang=["de"],
        )
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
                ),
            },
        )

        self.convert_pipe.add_component(
            "converter",
            DoclingConverter(
                converter=self.doc_converter,
                export_type=EXPORT_TYPE,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=512),
            ),
        )

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("converter", "file_saver")
    
    def init_easyocr_pipeline(self):
        """Initialize OCR pipeline with EasyOCR and PyPdfium backend."""
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options = EasyOcrOptions(
            force_full_page_ocr=True,
            lang=["de"],
        )
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                    backend=PyPdfiumDocumentBackend
                ),
            },
        )

        self.convert_pipe.add_component(
            "converter",
            DoclingConverter(
                converter=self.doc_converter,
                export_type=EXPORT_TYPE,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=512),
            ),
        )

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("converter", "file_saver")

    def init_tesseract_pipeline(self):
        #TODO remove deskew, put poppler path check here
        """Initialize OCR pipeline with deskewing preprocessing."""
        self.deskew_mode = True
        
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options = TesseractOcrOptions(
            force_full_page_ocr=True,
            lang=["deu"],
            psm=3,
        )
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                    backend=PyPdfiumDocumentBackend
                ),
            },
        )
        
        self.convert_pipe.add_component(
            "deskewer",
            Deskewer(self.preprocess_dir)
        )

        self.convert_pipe.add_component(
            "converter",
            DoclingConverter(
                converter=self.doc_converter,
                export_type=EXPORT_TYPE,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=512),
            ),
        )

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("deskewer", "converter")
        self.convert_pipe.connect("converter", "file_saver")
    
    def init_vlm_pipeline(self):
        """
        Initialize Vision Language Model pipeline.
        Reference: https://docling-project.github.io/docling/usage/vision_models/
        """
        self.pipeline_options = VlmPipelineOptions()
        self.pipeline_options.vlm_options = vlm_model_specs.SMOLDOCLING_TRANSFORMERS

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=self.pipeline_options,
                ),
            },
        )

        self.convert_pipe.add_component(
            "converter",
            DoclingConverter(
                converter=self.doc_converter,
                export_type=EXPORT_TYPE,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=512),
            ),
        )

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("converter", "file_saver")
    
    

    def convert_documents(self, paths: List[Path]):
        """
        Convert documents using the initialized pipeline.
        
        Args:
            paths: List of Path objects pointing to PDF files
            
        Returns:
            Pipeline execution result
        """
        res = self.convert_pipe.run({"converter": {"paths": paths}})
        return res