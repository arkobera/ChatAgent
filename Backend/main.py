import logging
from fastapi import FastAPI
from openai import OpenAI

import inngest #type: ignore
import inngest.fast_api #type: ignore

import uuid

from config import IS_PRODUCTION, get_service_setting
from data_loader import load_and_chunk_pdf_from_storage, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGQueryRequest, RAGQueryResult, RAGChunkAndSrc, RAGUpsertResult
inngest_client = inngest.Inngest(
    app_id = 'rag_app',
    logger = logging.getLogger('uvicorn'),
    is_production=IS_PRODUCTION,
    event_key=get_service_setting("INNGEST_EVENT_KEY"),
    signing_key=get_service_setting("INNGEST_SIGNING_KEY"),
    serializer = inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id='RAG: Ingest PDF',
    trigger=inngest.TriggerEvent(event='rag/ingest_pdf')
)
async def rag_ingest(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        storage_path = ctx.event.data['storage_path']
        source_id = ctx.event.data.get('source_id') or storage_path
        bucket = get_service_setting("SUPABASE_BUCKET", "pdfs")
        chunks = load_and_chunk_pdf_from_storage(bucket, storage_path) #type: ignore
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id) #type: ignore

    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunks
        source_id = chunk_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name= f"{source_id}:{i}")) for i in range(len(chunks))] 
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc) #type: ignore
    ingested = await ctx.step.run('emdeb-and-upsert', lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult) #type: ignore
    return ingested.model_dump()

def answer_question(question: str, top_k: int) -> RAGQueryResult:
    query_vec = embed_texts([question])[0]
    found = QdrantStorage().search(query_vec, top_k)
    context_block = "\n\n".join(f"- {context}" for context in found["contexts"])
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )
    client = OpenAI(
        api_key=get_service_setting("GROQ_API"),
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=1024,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You answer questions using only the provided context."},
            {"role": "user", "content": user_content},
        ],
    )
    answer = response.choices[0].message.content or ""
    return RAGQueryResult(
        answer=answer.strip(),
        sources=found["sources"],
        num_contexts=len(found["contexts"]),
    )

app = FastAPI()

@app.post("/api/query", response_model=RAGQueryResult)
def rag_query(query: RAGQueryRequest) -> RAGQueryResult:
    return answer_question(query.question, query.top_k)

inngest.fast_api.serve(app, inngest_client, [rag_ingest])
