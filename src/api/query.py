from fastapi import FastAPI
from pydantic import BaseModel
import supabase
from src.config.db_config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.ingestion.embedder import Embedder

app = FastAPI()

# Initialize Supabase client
sb = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

class QueryRequest(BaseModel):
    question: str
    document_ids: list[str] = None
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    limit: int = 10

@app.post("/query")
def query_rag(req: QueryRequest):
    # 1. Embed the question
    question_embedding = Embedder.encode([req.question])[0]

    # 2. Call hybrid_search SQL function
    doc_ids = req.document_ids or []
    results = sb.rpc(
        "hybrid_search",
        {
            "p_question": req.question,
            "p_question_embedding": question_embedding,
            "p_document_ids": doc_ids,
            "p_semantic_weight": req.semantic_weight,
            "p_keyword_weight": req.keyword_weight,
            "p_limit": req.limit
        }
    ).execute()

    # 3. Return structured response
    return {"results": results.data}
