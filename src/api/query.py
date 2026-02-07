from fastapi import APIRouter
from pydantic import BaseModel
from src.db.supabase import supabase

router = APIRouter()

# defines structure of incoming class adn validates it before executing business logic
class QueryRequest(BaseModel):
    question: str

@router.post("/query")
def query(req: QueryRequest):
    # TEMP sanity check
    res = supabase.table("documents").select("*").execute()
    return {
        "question": req.question,
        "documents_in_db": len(res.data)
    }