from openai import OpenAI #type: ignore
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
import requests
import os
import json
import tempfile
from pathlib import Path

from supabase import create_client, Client

from config import IS_PRODUCTION, LOCAL_UPLOAD_DIR, get_service_setting

JINA_API = get_service_setting('JINA_API_KEY')

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

supabase: Client | None = None
if IS_PRODUCTION:
    supabase = create_client(
        get_service_setting("SUPABASE_URL"), #type: ignore
        get_service_setting("SUPABASE_SERVICE_ROLE_KEY") #type: ignore
    )

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path) #type: ignore
    texts = [d.text for d in docs if getattr(d,'text',None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def load_and_chunk_pdf_from_storage(bucket: str, storage_path: str):
    if not IS_PRODUCTION:
        local_path = (LOCAL_UPLOAD_DIR.parent / storage_path).resolve()
        if not local_path.is_relative_to(LOCAL_UPLOAD_DIR.resolve()):
            raise ValueError("Local upload path must be inside the uploads directory.")
        if not local_path.is_file():
            raise FileNotFoundError(f"Local upload not found: {local_path}")
        return load_and_chunk_pdf(str(local_path))

    if supabase is None:
        raise RuntimeError("Supabase storage is only available when APP_ENV=PROD.")
    pdf_bytes = supabase.storage.from_(bucket).download(storage_path)
    file_descriptor, temporary_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(pdf_bytes)
        return load_and_chunk_pdf(temporary_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

def embed_texts(texts: list[str])-> list[list[float]]:
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {JINA_API}"
    }
    data={
        "model":"jina-embeddings-v5-text-small",
        "task":"retrieval.query",
        "normalized":True,
        "input":texts
    }
    response = requests.post(url,headers=headers, data=json.dumps(data))
    response.raise_for_status()
    data = response.json()
    # return [item['embedding'] for item in data['data']]
    return [item["embedding"] for item in data["data"]]

