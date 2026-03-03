import os
from pathlib import Path
from docling_haystack.converter import DoclingConverter, ExportType
from haystack import Pipeline, Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from docling.chunking import HybridChunker

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions,
    EasyOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption, ConversionResult
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from haystack.document_stores.types import DuplicatePolicy

from haystack import component

from haystack.utils import Secret

from collections import defaultdict

from docling.pipeline.simple_pipeline import SimplePipeline

import json

from typing import List, Dict, Literal, Optional
from pydantic import BaseModel
from datetime import date

import logging

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-large-instruct")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "360"))
EXPORT_TYPE = ExportType.DOC_CHUNKS

# pydantic metadata model
class ReportMetadata(BaseModel):
    patient_id: Optional[str]
    entlassungsdatum: Optional[date]
    aufnahmedatum: Optional[date]
    diagnosen: Optional[list[str]]
    vordiagnosen: Optional[list[str]]
    icd_codes: Optional[list[str]]
    medikation: Optional[list[str]]
    eingriffe: Optional[list[str]]


@component
class MetadataExtractor:
    # https://docs.haystack.deepset.ai/docs/llmmetadataextractor exists, but does not support OllamaGenerator (Jan 2026)
    # https://haystack.deepset.ai/blog/extracting-metadata-filter
    def __init__(self):
        self.system_prompt = ChatMessage.from_system("""
            Du bist ein System zur Extraktion medizinischer Informationen aus Befundtexten und Arztbriefen. Extrahiere strukturierte Daten aus diesem Befundtext.

            Der OCR-Text kann Fehler, fehlende Informationen oder Formatierungsprobleme enthalten. Gib dein Bestes, um so viel wie möglich zu extrahieren.

            Extrahiere die folgenden Informationen im JSON-Format:
            {
            "patient_id": "string oder null",
            "entlassungsdatum": "YYYY-MM-DD oder null",
            "aufnahmedatum": "YYYY-MM-DD oder null",
            "diagnosen": ["Liste der Diagnosen"],
            "vordiagnosen": ["Liste der vorherigen Diagnosen"],
            "icd_codes": ["Liste der ICD-10 codes falls präsent"],
            "medikation": ["Liste der Medikamente"],
            "eingriffe": ["Liste der durchgeführten Eingriffe"],
            }

            Wichtig:
            - Es gibt meistens ein bis zwei Diagnosen.
            - Es gibt meistens mehrere Vordiagnosen. Manchmal stehen sie in einem gesonderten Abschnitt, manchmal werden sie mit "Z.n." (Zustand nach) gekennzeichnet.
            - Verwende null für alle Felder, die du nicht finden kannst
            - Konvertiere Datumsangaben in das Format JJJJ-MM-TT


            Gib nur JSON als Antwort ohne zusätzliche Erklärungen oder Text.
        """)
        self.user_prompt = ChatMessage.from_user("""
            Befundtext:
            {{ document_text }}
        """)

        self.pipeline = Pipeline()
        self.builder = ChatPromptBuilder()
        self.llm = OllamaChatGenerator(
            url=OLLAMA_URL,
            model=OLLAMA_MODEL,
            timeout=OLLAMA_TIMEOUT,
            response_format=ReportMetadata.model_json_schema(),
            generation_kwargs={
                "temperature": OLLAMA_TEMPERATURE,
                "num_ctx": OLLAMA_NUM_CTX,
            }
        )
        self.pipeline.add_component(name="builder", instance=self.builder)
        self.pipeline.add_component(name="llm", instance=self.llm)
        self.pipeline.connect("builder.prompt", "llm.messages")
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        #TODO: document chunks wieder auf ganze dokumente mergen
        results = []
        for doc in documents:
            result = self.pipeline.run(
                {
                    "builder": {
                        "template_variables": {
                             "document_text": doc.content,
                        },
                        "template": [self.system_prompt, self.user_prompt],
                    }
                }
            )
            response = result["llm"]["replies"][0]
            logger.info(f"ChatMessage result dir: {dir(response)}")
            try:
                logger.info(f"LLM response for document {doc.meta.get('file_name', 'unknown')}: {response.text}")
                # remove leading and trailing text around the JSON
                #response_text = response_text[response_text.find("{"):]
                #response_text = response_text[:response_text.rfind("}")+1] 
                metadata = json.loads(response.text)
            except json.JSONDecodeError:
                print("JSON decode error, setting metadata to default nulls")
                metadata = json.loads('{"patient_id": null, "entlassungsdatum": null, "aufnahmedatum": null, "diagnosen": [], "icd_codes": [], "medikation": [], "eingriffe": []}')
            results.append(Document(
                content=doc.content,
                meta={**doc.meta, "extracted_metadata": metadata}
            ))
        return {"documents": results}
    # @component.output_types(metadata=ReportMetadata)
    # def run(self, document_text: str):
    #     results = self.pipeline.run(
    #         {
    #             "prompt_builder": {"document_text": document_text}
    #         }
    #     )
    #     response_text = results["llm"]["replies"][0]
    #     try:
    #         metadata = json.loads(response_text)
    #     except json.JSONDecodeError:
    #         metadata = json.loads('{"patient_id": null, "entlassungsdatum": null, "aufnahmedatum": null, "diagnosen": [], "icd_codes": [], "medikation": [], "eingriffe": [], "konfidenz": {}}')
        
    #     return metadata
        # filters = []
        # for key, value in metadata.items():
        #     field = f"meta.{key}"
        #     filters.append({f"field": field, "operator": "==", "value": value})

        # return {"filters": {"operator": "AND", "conditions": filters}}

class DoclingIndexerService:
    
    def __init__(self):
        self.document_store = None
        self.idx_pipe = None
        self.entry_component = None
        self.available_models = {
            "intfloat/multilingual-e5-large-instruct" : {"embedding_dimension": 1024, "prefix" : ""},
            "Qwen/Qwen3-Embedding-0.6B": {"embedding_dimension": 1024, "prefix" : ""},
        }
        self.set_document_store(document_store=None)
    
    def set_document_store(self, document_store: PgvectorDocumentStore = None, connection_string: str = None, embedding_dimension: int = None, language: str = None, vector_function: str = None, recreate_table: bool = False, search_strategy: str = None):
        if document_store is not None:
            print("Using provided document store")
            print("###MAKE SURE THIS IS INTENDED AND COMPATIBLE WITH THE CHOSEN HUGGINGFACE EMBEDDING MODEL###")
            self.document_store = document_store
        else:
            print("Setting custom document store")
            self.document_store = PgvectorDocumentStore(
                connection_string=Secret.from_env_var("PG_CONN_STR") if connection_string is None else Secret.from_token(connection_string),
                embedding_dimension=self.available_models.get(EMBED_MODEL_ID, {"embedding_dimension": 1024})["embedding_dimension"] if embedding_dimension is None else embedding_dimension,
                language="german" if language is None else language,
                vector_function="cosine_similarity" if vector_function is None else vector_function,
                recreate_table=True if recreate_table is None else recreate_table,
                search_strategy="exact_nearest_neighbor" if search_strategy is None else search_strategy,
            )
    
    def init_simple_pipeline(self):
        logger.info("Initializing simple indexing pipeline")
        self.entry_component = "embedder"
        self.idx_pipe = Pipeline()

        self.idx_pipe.add_component(
            "embedder",
            SentenceTransformersDocumentEmbedder(
                model=EMBED_MODEL_ID,
                prefix=self.available_models.get(EMBED_MODEL_ID, {"prefix": ""})["prefix"],
                tokenizer_kwargs={"truncation": True, "max_length": 512}
            ),
        )

        self.idx_pipe.add_component(
            "writer",
            DocumentWriter(
                document_store=self.document_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        )

        self.idx_pipe.connect("embedder", "writer")

    def init_metadata_extractor_pipeline(self):
        logger.info("Initializing metadata extractor pipeline")
        self.entry_component = "metadata_extractor"
        self.idx_pipe = Pipeline()

        self.idx_pipe.add_component(
            "metadata_extractor",
            MetadataExtractor()
        )

        self.idx_pipe.add_component(
            "embedder",
            SentenceTransformersDocumentEmbedder(
                model=EMBED_MODEL_ID,
                prefix=self.available_models.get(EMBED_MODEL_ID, {"prefix": ""})["prefix"],
                tokenizer_kwargs={"truncation": True, "max_length": 512}
            ),
        )

        self.idx_pipe.add_component(
            "writer",
            DocumentWriter(
                document_store=self.document_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        )
        self.idx_pipe.connect("metadata_extractor", "embedder")
        self.idx_pipe.connect("embedder", "writer")
        

    def index_documents(self, input_documents: list[Document] = None):
        self.idx_pipe.run({self.entry_component: {"documents": input_documents}})