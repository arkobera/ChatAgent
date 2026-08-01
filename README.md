# ChatAgent
An agent that connects with your whatsapp and chats on your behalf

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

