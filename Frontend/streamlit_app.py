from pathlib import Path
from uuid import uuid4

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
from supabase import Client, create_client
from Authentication.auth import Auth, AuthScreen

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "LOCAL").strip().upper()
if APP_ENV not in {"LOCAL", "PROD"}:
    raise RuntimeError("APP_ENV must be either 'LOCAL' or 'PROD'.")


def get_service_setting(name: str, default: str | None = None) -> str | None:
    return os.getenv(f"{name}_{APP_ENV}") or os.getenv(name, default)


IS_PRODUCTION = APP_ENV == "PROD"
SUPABASE_BUCKET = get_service_setting("SUPABASE_BUCKET", "pdfs")
LOCAL_UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
BACKEND_API_URL = (
    get_service_setting("RENDER_API", "") if IS_PRODUCTION
    else get_service_setting("BACKEND_API_URL", "http://127.0.0.1:8000")
).rstrip("/") #type: ignore
supabase: Client | None = None
if IS_PRODUCTION:
    supabase = create_client(
        get_service_setting("SUPABASE_URL"),  # type: ignore
        get_service_setting("SUPABASE_KEY"),  # type: ignore
    )

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


def require_authentication() -> None:
    """Render the sign-in page and stop before protected app content."""
    if not st.session_state.get("user_email"):
        AuthScreen().auth_screen()
        st.stop()


def render_logout() -> None:
    with st.sidebar:
        st.caption(f"Signed in as {st.session_state.user_email}")
        if st.button("Sign out"):
            error = Auth().sign_out()
            if error:
                st.error(f"Sign out failed: {error}")
                return
            st.session_state.pop("user_email", None)
            st.rerun()


require_authentication()
render_logout()


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        is_production=IS_PRODUCTION,
        event_key=get_service_setting("INNGEST_EVENT_KEY"),
        signing_key=get_service_setting("INNGEST_SIGNING_KEY"),
    )


def upload_pdf(file) -> str:
    filename = Path(file.name).name
    storage_path = f"uploads/{uuid4()}-{filename}"
    if not IS_PRODUCTION:
        LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        (LOCAL_UPLOAD_DIR / Path(storage_path).name).write_bytes(file.getvalue())
        return storage_path

    if supabase is None:
        raise RuntimeError("Supabase storage is only available when APP_ENV=PROD.")
    supabase.storage.from_(SUPABASE_BUCKET).upload( #type: ignore
        path=storage_path,
        file=file.getvalue(),
        file_options={"content-type": "application/pdf", "upsert": "false"},
    )
    return storage_path


def send_rag_ingest_event(storage_path: str, source_id: str) -> None:
    client = get_inngest_client()
    client.send_sync(
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
        storage_path = upload_pdf(uploaded)
        # st.status(storage_path)///
        send_rag_ingest_event(storage_path, uploaded.name)

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
