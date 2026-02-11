import os
from pathlib import Path
from docling_haystack.converter import DoclingConverter, ExportType
from haystack import Pipeline, Document

from docling.chunking import HybridChunker

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions,
    EasyOcrOptions,
    VlmPipelineOptions,
    RapidOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption, ConversionResult
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from docling.pipeline.vlm_pipeline import VlmPipeline

from haystack import component

from collections import defaultdict

from docling.datamodel import vlm_model_specs

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

from typing import Iterable, Union

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-large-instruct")
EXPORT_TYPE = ExportType.DOC_CHUNKS


@component
class TextFileSaver:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        grouped = defaultdict(list)

        for doc in documents:
            try:
                #TODO this doesn't work for the VLM pipeline
                source = doc.meta["dl_meta"]["meta"]["origin"]["filename"]
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
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def deskew(self, img):
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Deskew
        coords = np.column_stack(np.where(gray < 200))
        if len(coords) > 0:
            #angle = cv2.minAreaRect(np.column_stack((coords[:,1],coords[:,0])))[-1]
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            print(angle)
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

    #https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
    # Calculate skew angle of an image
    def getSkewAngle(self, img) -> float:
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
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        # Find largest contour and surround in min area box
        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)

        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    # Rotate the image around its center
    def rotateImage(self, img, angle: float):
        newImage = np.array(img)
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(newImage)

    # Deskew image
    def deskew_old(self, img):
        angle = self.getSkewAngle(img)
        return self.rotateImage(img, -1.0 * angle)
    
    #TODO try ocrmypdf https://stackoverflow.com/questions/41546181/how-to-deskew-a-scanned-text-page-with-imagemagick
    @component.output_types(paths=Iterable[Union[Path, str]])
    def run(self, paths: list[Path]):
        output_paths = []
        for path in paths:
            processed_pages = []
            pages = convert_from_path(path, 400, poppler_path="/opt/homebrew/bin")
            for page in pages:
                processed_pages.append(self.deskew(page))
            output_path = self.output_dir / Path(path).name
            processed_pages[0].save(
                output_path, 
                save_all=True, 
                append_images=processed_pages[1:],
                resolution=400.0
            )
            output_paths.append(output_path)
        return {"paths": output_paths}

class DoclingConvertingService:
    
    def __init__(self, preprocess_dir: Path = None, output_dir: Path = None, rapidocr_models_path: Path = None):
        self.preprocess_dir = preprocess_dir
        self.output_dir = output_dir
        self.rapidocr_models_path = rapidocr_models_path

        self.convert_pipe = Pipeline()

        self.pipeline_options = None

        self.doc_converter = None


    def init_ocr_pipeline(self):
        # self.pipeline_options = PdfPipelineOptions()
        # self.pipeline_options.do_ocr = True
        # self.pipeline_options.ocr_options.force_full_page_ocr = True
        #self.pipeline_options.ocr_options = TesseractOcrOptions(
        #    force_full_page_ocr = True,
        #    lang=["deu"],
        #    psm=3,
        #)
        # self.pipeline_options.ocr_options = EasyOcrOptions(
        #    force_full_page_ocr = True,
        #    lang=["de"],
        # )
        #self.pipeline_options.do_table_structure = True
        #self.pipeline_options.table_structure_options.do_cell_matching = True

        det_model_path = os.path.join(
            self.rapidocr_models_path, "onnx", "PP-OCRv4", "det", "en_PP-OCRv3_det_infer.onnx"
        )
        rec_model_path = os.path.join(
            self.rapidocr_models_path, "onnx", "PP-OCRv5", "rec", "latin_PP-OCRv5_rec_mobile_infer.onnx"
        )
        cls_model_path = os.path.join(
            self.rapidocr_models_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        )
        ocr_options = RapidOcrOptions(
            force_full_page_ocr=True,
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            cls_model_path=cls_model_path,
        )
        self.pipeline_options = PdfPipelineOptions(
            ocr_options=ocr_options,
        )

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                    #backend=PyPdfiumDocumentBackend
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

    def init_ocr_deskew_pipeline(self):
        self.deskew_mode = True
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options = TesseractOcrOptions(
            force_full_page_ocr = True,
            lang=["deu"],
            psm=3,
        )
        #self.pipeline_options.ocr_options = EasyOcrOptions(
        #    force_full_page_ocr = True,
        #    lang=["de"],
        #)
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
        #TODO document cleaner?
        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("deskewer", "converter")
        self.convert_pipe.connect("converter", "file_saver")
    
    def init_vlm_pipeline(self):
        #https://docling-project.github.io/docling/usage/vision_models/
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
        #TODO: preprocessors? deskewing?
        self.convert_pipe.add_component(
            "converter",
            DoclingConverter(
                converter=self.doc_converter,
                export_type=EXPORT_TYPE,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=512),
            ),
        )
        #TODO document cleaner?
        self.convert_pipe.add_component(
            "file_saver",
            TextFileSaver(output_dir=self.output_dir),
        )

        self.convert_pipe.connect("converter", "file_saver")

    def convert_documents(self, paths: list[Path]):
        res = self.convert_pipe.run({"converter": {"paths": paths}})
        return res