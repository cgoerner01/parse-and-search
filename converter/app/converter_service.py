"""
Docling Converting Service - extracted from original code for use with FastAPI.
"""
from abc import ABC, abstractmethod
import os
from pathlib import Path
from docling_haystack.converter import DoclingConverter, ExportType
from haystack import Pipeline, Document
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack import component
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage, ImageContent, FileContent
from haystack.components.converters.image import PDFToImageContent

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

import httpx
import base64

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.DEBUG)

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-large-instruct")
EXPORT_TYPE = ExportType.DOC_CHUNKS
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
#OLLAMA_MODEL_OCR = os.getenv("OLLAMA_MODEL_OCR", "deepseek-ocr")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "360"))

@component
class TextFileSaver:
    """Component to save document chunks to text files."""
    
    def __init__(self, output_dir: str):
        self.output_dir = str(Path(output_dir))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

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
            out_path = Path(self.output_dir) / f"{name}.txt"

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(contents))

        return {"documents": documents}


@component
class Deskewer:
    """Component to deskew PDF pages before OCR processing."""
    
    def __init__(self, output_dir: str):
        self.output_dir = str(Path(output_dir))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
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

    def deskew_hough(self, img):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_cv, 50, 150, apertureSize=3)
        # Look for lines that represent text rows
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Ignore purely vertical or extreme lines
            if abs(angle) < 45:
                angles.append(angle)
                
        return np.median(angles) if angles else 0

    def rotate_image(self, pil_img, angle):
        """Rotates image and adds white padding to prevent data loss."""
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        (h, w) = img_cv.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box to prevent cropping corners
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate with white background
        rotated = cv2.warpAffine(img_cv, M, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
        
        return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    
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
                #processed_pages.append(self.deskew(page))
                processed_pages.append(self.rotate_image(page, self.deskew_hough(page)))
            
            # Save deskewed PDF
            output_path = Path(self.output_dir) / Path(path).name
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
class PdfPrerenderer:
    """
    Re-renders every PDF page as a bitmap image before Docling sees it.
    Forces Docling to treat all pages as image-only, preventing silent skips.
    """
    
    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = str(Path(output_dir))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    @component.output_types(paths=Iterable[Union[Path, str]])
    def run(self, paths: List[Path]):
        output_paths = []
        
        for path in paths:
            output_path = Path(self.output_dir) / f"prerendered_{path.name}"
            
            src = fitz.open(path)
            dst = fitz.open()
            
            for page in src:
                pix = page.get_pixmap(dpi=self.dpi, colorspace=fitz.csRGB)
                img_page = dst.new_page(width=pix.width, height=pix.height)
                img_page.insert_image(img_page.rect, pixmap=pix)
            
            dst.save(output_path, deflate=True)
            src.close()
            dst.close()
            
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
        self.preprocess_dir = str(Path(preprocess_dir))
        self.predictors = load_predictors()
    
    @component.output_types(documents=list[Document])
    def run(self, paths: Iterable[Union[Path, str]]):
        results = []
        for path in paths:
            pages = []
            with fitz.open(path) as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=200)
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

# Remove OllamaOCRConverter base class entirely.
# Inline the shared `run()` and `encode_image()` logic as a mixin or helper function.

def _encode_image(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _ollama_ocr_run(self, paths):
    """Shared run logic for Ollama OCR converters."""
    results = []
    for path in paths:
        pages = []
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_save_path = Path(self.preprocess_dir) / f"page_{Path(path).stem}_page{page.number+1}.jpg"
                img.save(page_save_path)
                res = self.ocr_image(str(page_save_path), mode="Text", host=OLLAMA_URL)
                pages.append(res)
        results.append(pages)

    return {"documents": [
        Document(
            content='\n'.join(res),
            meta={"source": str(Path(path).name)},
        )
        for res, path in zip(results, paths)
    ]}


@component
class OllamaGLMOCRConverter:
    def __init__(self, preprocess_dir: str = None, model_name: str = "glm-ocr"):
        self.preprocess_dir = str(Path(preprocess_dir))
        self.model_name = model_name

    def encode_image(self, image_path):
        return _encode_image(image_path)

    def ocr_image(self, image_path, mode="Text", host=OLLAMA_URL):
        image_data, _ = self.encode_image(image_path)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": f"{mode} Recognition:", "images": [image_data]}],
            "stream": False,
            "options": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.3, "num_ctx": 32000},
        }
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            response = client.post(f"{host}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]

    @component.output_types(documents=list[Document])
    def run(self, paths: Iterable[Union[Path, str]]):
        return _ollama_ocr_run(self, paths)


@component
class OllamaDeepSeekOCRConverter:
    def __init__(self, preprocess_dir: str = None, model_name: str = "deepseek-ocr"):
        self.preprocess_dir = str(Path(preprocess_dir))
        self.model_name = model_name

    def encode_image(self, image_path):
        return _encode_image(image_path)

    def ocr_image(self, image_path, mode="text", host=OLLAMA_URL):
        image_data, _ = self.encode_image(image_path)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "\nFree OCR.", "images": [image_data]}],
            "stream": False,
            "options": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.3},
        }
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            response = client.post(f"{host}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]

    @component.output_types(documents=list[Document])
    def run(self, paths: Iterable[Union[Path, str]]):
        return _ollama_ocr_run(self, paths)

# class OllamaOCRConverter:
#     """Component to convert PDFs using Ollama OCR."""
    
#     def __init__(self, preprocess_dir: str = None, model_name: str = "glm-ocr"):
#         self.preprocess_dir = str(Path(preprocess_dir))
#         self.model_name = model_name

#     def encode_image(self, image_path: str) -> tuple[str, str]:
#         path = Path(image_path)
#         suffix = path.suffix.lower()
#         mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix, "image/jpeg")
#         with open(path, "rb") as f:
#             return base64.b64encode(f.read()).decode(), mime

#     def ocr_image(self, *args, **kwargs):
#         raise NotImplementedError("Subclasses must implement ocr_image()")

#     def run(self, paths: Iterable[Union[Path, str]]):
#         results = []
#         for path in paths:
#             pages = []
#             with fitz.open(path) as doc:
#                 for page in doc:
#                     pix = page.get_pixmap(dpi=150)
#                     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                     # save img to preprocess_dir
#                     page_save_path = Path(self.preprocess_dir) / f"page_{Path(path).stem}_page{page.number+1}.jpg"
#                     img.save(page_save_path)
#                     # call ollama API
#                     res = self.ocr_image(str(page_save_path), mode="Text", host=OLLAMA_URL)
#                     pages.append(res)
#                 results.append(pages)

#         output_docs = []
#         for res, path in zip(results, paths):
#             output_docs.append(
#                 Document(
#                     content= '\n'.join(res),
#                     meta={"source": str(Path(path).name)},
#                 )
#             )
#         return {"documents": output_docs}

# @component
# class OllamaGLMOCRConverter(OllamaOCRConverter):
#     """Component to convert PDFs using Ollama GLM-OCR."""
    
#     def __init__(self, preprocess_dir: str = None, model_name: str = "glm-ocr"):
#         super().__init__(preprocess_dir, model_name=model_name)

#     def ocr_image(
#         self,
#         image_path: str,
#         mode: str = "Text",  # "Text", "Table", or "Figure"
#         host: str = "http://ollama:11434",
#     ) -> str:
#         image_data, mime_type = self.encode_image(image_path)

#         payload = {
#             "model": self.model_name,
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": f"{mode} Recognition:",
#                     "images": [image_data],  # base64, no data URI prefix
#                 }
#             ],
#             "stream": False,
#             "options": {
#                 "temperature": 0.0,
#                 "top_k": 1,
#                 "top_p": 1.0,
#                 "repeat_penalty": 1.3,
#                 "num_ctx": 32000,
#             },
#         }

#         # Fresh client per call = no connection/session state leaking over
#         with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
#             response = client.post(f"{host}/api/chat", json=payload)
#             response.raise_for_status()
#             return response.json()["message"]["content"]
    
#     @component.output_types(documents=list[Document])
#     def run(self, paths: Iterable[Union[Path, str]]):
#         return super().run(paths)

# @component
# class OllamaDeepSeekOCRConverter(OllamaOCRConverter):
#     """Component to convert PDFs using Ollama DeepSeek-OCR."""
    
#     def __init__(self, preprocess_dir: str = None, model_name: str = "deepseek-ocr"):
#         super().__init__(preprocess_dir, model_name=model_name)

#     def ocr_image(
#         self,
#         image_path: str,
#         mode: str = "text",  # "text", "markdown", "layout", "figure", "extract"
#         host: str = "http://ollama:11434",
#     ) -> str:
#         image_data, _ = self.encode_image(image_path)

#         payload = {
#             "model": self.model_name,
#             "messages": [
#                 {
#                     "role": "user",
#                     # DeepSeek-OCR expects: image + "\n" + instruction
#                     "content": "\nFree OCR.",
#                     "images": [image_data],
#                 }
#             ],
#             "stream": False,
#             "options": {
#                 "temperature": 0.0,
#                 "top_k": 1,
#                 "top_p": 1.0,
#                 "repeat_penalty": 1.3,
#             },
#         }

#         with httpx.Client(timeout=120.0) as client:
#             response = client.post(f"{host}/api/chat", json=payload)
#             response.raise_for_status()
#             return response.json()["message"]["content"]

#     @component.output_types(documents=list[Document])
#     def run(self, paths: Iterable[Union[Path, str]]):
#         return super().run(paths)


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
    
    def init_ollama_glm_ocr_pipeline(self):
        """Initialize OCR pipeline with Ollama DeepSeek OCR."""
        self.convert_pipe.add_component(
            "deskewer",
            Deskewer(self.preprocess_dir)
        )

        self.convert_pipe.add_component(
            "converter",
            OllamaGLMOCRConverter(preprocess_dir=self.preprocess_dir, model_name="glm-ocr"),
        )

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("deskewer", "converter")
        self.convert_pipe.connect("converter", "file_saver")

    def init_ollama_deepseek_ocr_pipeline(self):
        """Initialize OCR pipeline with Ollama DeepSeek OCR."""
        self.convert_pipe.add_component(
            "deskewer",
            Deskewer(self.preprocess_dir)
        )

        self.convert_pipe.add_component(
            "converter",
            OllamaDeepSeekOCRConverter(preprocess_dir=self.preprocess_dir, model_name="deepseek-ocr"),
        )

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("deskewer", "converter")
        self.convert_pipe.connect("converter", "file_saver")

    def init_rapidocr_pipeline(self):
        """Initialize OCR pipeline with RapidOCR."""
        
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
            do_table_structure=False,
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
            "deskewer",
            Deskewer(self.preprocess_dir)
        )

        #self.convert_pipe.add_component(
        #    "prerenderer",
        #    PdfPrerenderer(output_dir=self.preprocess_dir, dpi=300),
        #)

        self.convert_pipe.add_component(
            "converter",
            DoclingConverter(
                converter=self.doc_converter,
                export_type=EXPORT_TYPE,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=512),
            ),
        )
        

        #self.convert_pipe.add_component(
        #    "converter",
        #    RapidOCRConverter(preprocess_dir=self.preprocess_dir),
        #)

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
        self.convert_pipe.connect("deskewer", "converter")
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
            "deskewer",
            Deskewer(self.preprocess_dir)
        )

        self.convert_pipe.add_component(
            "converter",
            SuryaOCRConverter(preprocess_dir=self.preprocess_dir),
        )

        #self.convert_pipe.add_component(
        #    "cleaner",
        #    DocumentCleaner(
        #        remove_substrings=[
        #            "999999 Elektronisches Dokument",
        #            "Keine Archivierung des Ausdrucks am UKHD"
        #        ],
        #        replace_regexes={
        #            r'Ausdruck aus.* Unterbelegart:.*(riefe|richte)' : '',
        #            r'Gedruckte Seiten.*Datum: \d{2}\.\d{2}\.\d{4}' : '',
        #        }
        #    ),
        #)

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("deskewer", "converter")
        self.convert_pipe.connect("converter", "file_saver")
        #self.convert_pipe.connect("cleaner", "file_saver")
    
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

        #self.convert_pipe.add_component(
        #    "cleaner",
        #    DocumentCleaner(
        #        remove_substrings=[
        #            "999999 Elektronisches Dokument",
        #            "Keine Archivierung des Ausdrucks am UKHD"
        #        ],
        #        replace_regexes={
        #            r'Ausdruck aus.* Unterbelegart:.*(riefe|richte)' : '',
        #            r'Gedruckte Seiten.*Datum: \d{2}\.\d{2}\.\d{4}' : '',
        #        }
        #    ),
        #)

        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("deskewer", "converter")
        self.convert_pipe.connect("converter", "file_saver")
    
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
    
    

    def convert_documents(self, paths: List[Path]):
        """
        Convert documents using the initialized pipeline.
        
        Args:
            paths: List of Path objects pointing to PDF files
            
        Returns:
            Pipeline execution result
        """
        #res = self.convert_pipe.run({"converter": {"paths": paths}})
        # if deskewer not in pipeline
        if "deskewer" not in list(self.convert_pipe.to_dict()['components'].keys()):
            res = self.convert_pipe.run({"converter": {"paths": paths}})
        else:
            res = self.convert_pipe.run({"deskewer": {"paths": paths}})
        return res