import asyncio
from pathlib import Path
from uuid import uuid4

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
from supabase import Client, create_client

load_dotenv()

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "pdfs")
BACKEND_API_URL = os.getenv("RENDER_API", "").rstrip("/")
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"),  # type: ignore
    os.environ.get("SUPABASE_KEY"),  # type: ignore
)

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app")


def upload_pdf_to_supabase(file) -> str:
    filename = Path(file.name).name
    storage_path = f"uploads/{uuid4()}-{filename}"
    supabase.storage.from_(SUPABASE_BUCKET).upload(
        path=storage_path,
        file=file.getvalue(),
        file_options={"content-type": "application/pdf", "upsert": "false"},
    )
    return storage_path


async def send_rag_ingest_event(storage_path: str, source_id: str) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "storage_path": storage_path,
                "source_id": source_id,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        storage_path = upload_pdf_to_supabase(uploaded)
        st.status(storage_path)
        asyncio.run(send_rag_ingest_event(storage_path, uploaded.name))
    st.success(f"Uploaded and triggered ingestion for: {uploaded.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


def query_backend(question: str, top_k: int) -> dict:
    if not BACKEND_API_URL:
        raise RuntimeError("RENDER_API must be set to your Render service URL.")
    response = requests.post(
        f"{BACKEND_API_URL}/api/query",
        json={"question": question, "top_k": top_k},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Searching your documents and generating an answer..."):
            output = query_backend(question.strip(), int(top_k))
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")