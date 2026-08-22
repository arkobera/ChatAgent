# ChatAgent
An agent that connects with your whatsapp and chats on your behalf

Set `APP_ENV` in `.env` to either `LOCAL` or `PROD`. Service settings can be
scoped with a suffix such as `QDRANT_URL_LOCAL` or `QDRANT_URL_PROD`; the
existing un-suffixed setting remains a fallback for cloud services. LOCAL uses
`http://127.0.0.1:6333` for Qdrant and `http://127.0.0.1:8000` for the backend
unless those scoped values are provided. PDF uploads are saved to the local
`uploads/` directory in LOCAL and use Supabase Storage only in PROD.

Terminal 1 (Backend)
```
uv run uvicorn main:app
```

Terminal 2 (Inngest)
```
npx inngest-cli@latest dev -u "http://127.0.0.1:8000/api/inngest" -no-discovery
```

Terminal 3 (Qdrant Local)
```
docker run -d --name qdrantDB -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

