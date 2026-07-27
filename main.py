import logging
from fastapi import FastAPI

import inngest #type: ignore
import inngest.fast_api #type: ignore
from inngest.experimental import ai #type: ignore

from dotenv import load_dotenv

import uuid
import os
import datetime

load_dotenv()

inngest_client = inngest.Inngest(
    app_id = 'rap_app',
    logger = logging.getLogger('uvicorn'),
    is_production = False,
    serializer = inngest.PydanticSerializer()
)

app = FastAPI()

@inngest_client.create_function(
    fn_id='RAG: Ingest PDF',
    trigger=inngest.TriggerEvent(event='rag/ingest_pdf')
)
async def rag_ingest(ctx: inngest.Context):
    return {"heallo":"world"}

inngest.fast_api.serve(app, inngest_client, [rag_ingest])