import os
from pathlib import Path
from docling_haystack.converter import DoclingConverter, ExportType
from haystack import Pipeline, Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
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

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-large-instruct")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
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
    konfidenz: Optional[Dict[str, Literal["hoch", "medium", "niedrig"]]]


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
class MetadataExtractor:
    # https://docs.haystack.deepset.ai/docs/llmmetadataextractor exists, but does not support OllamaGenerator (Jan 2026)
    # https://haystack.deepset.ai/blog/extracting-metadata-filter
    def __init__(self):
        self.prompt = """
            Du bist ein System zur Extraktion medizinischer Informationen. Extrahiere strukturierte Daten aus diesem Befundtext.

            Der OCR-Text kann Fehler, fehlende Informationen oder Formatierungsprobleme enthalten. Gib dein Bestes, um so viel wie möglich zu extrahieren.

            Befundtext:
            {{ document_text }}

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
            "konfidenz": {
                "entlassungsdatum": "hoch/medium/niedrig",
                "diagnosen": "hoch/medium/niedrig"
            }
            }

            Wichtig:
            - Verwende null für alle Felder, die du nicht finden kannst
            - Konvertiere Datumsangaben in das Format JJJJ-MM-TT
            - Füge Konfidenzstufen für kritische Felder hinzu
            - Wenn der Text aufgrund von OCR-Fehlern unklar ist, markiere die Konfidenz als "niedrig"

            Gib nur JSON als Antwort ohne zusätzliche Erklärungen oder Text.
        """

        self.pipeline = Pipeline()
        self.builder = PromptBuilder(self.prompt)
        self.llm = OllamaGenerator(
            url=OLLAMA_URL,
            model=OLLAMA_MODEL,
            generation_kwargs={
                "format": json.dumps(ReportMetadata.model_json_schema()),
            }
        )
        self.pipeline.add_component(name="builder", instance=self.builder)
        self.pipeline.add_component(name="llm", instance=self.llm)
        self.pipeline.connect("builder", "llm")
    
    @component.output_types(metadata=List[Document])
    def run(self, documents: List[Document]):
        #TODO: document chunks wieder auf ganze dokumente mergen
        results = []
        for doc in documents:
            result = self.pipeline.run(
                {
                    "builder": {"document_text": doc.content}
                }
            )
            response_text = result["llm"]["replies"][0]
            try:
                metadata = json.loads(response_text)
            except json.JSONDecodeError:
                print("JSON decode error, setting metadata to default nulls")
                metadata = json.loads('{"patient_id": null, "entlassungsdatum": null, "aufnahmedatum": null, "diagnosen": [], "icd_codes": [], "medikation": [], "eingriffe": [], "konfidenz": {}}')
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

class DoclingIndexingService:
    
    def __init__(self, input_documents: list[Document], document_store: PgvectorDocumentStore = None):
        self.input_documents = input_documents
        self.document_store = document_store or PgvectorDocumentStore(
            connection_string=Secret.from_env_var("PG_CONN_STR"),
            embedding_dimension=1024,
            language="german",
            vector_function="cosine_similarity",
            recreate_table=True,
            search_strategy="exact_nearest_neighbor",
        )

        self.idx_pipe = Pipeline()

        self.idx_pipe.add_component(
            "metadata_extractor",
            MetadataExtractor()
        )

        self.idx_pipe.add_component(
            "embedder",
            SentenceTransformersDocumentEmbedder(
                model=EMBED_MODEL_ID,
                #TODO nochmal nachlesen
                prefix="",
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
        

    def index_documents(self):
        # Extract metadata, embed, and write to document store
        self.idx_pipe.run({"metadata_extractor": {"documents": self.input_documents}})