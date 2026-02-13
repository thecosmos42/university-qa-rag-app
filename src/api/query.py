from fastapi import APIRouter
from pydantic import BaseModel
import supabase
from src.config.db_config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.ingestion.embedder import Embedder
from src.api.RAGGenerator import RAGGenerator

router = APIRouter()
emb = Embedder()

# Initialize Supabase client
sb = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

class QueryRequest(BaseModel):
    question: str
    document_ids: list[str] = None
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    limit: int = 10

@router.post("/query")
def query_rag(req: QueryRequest):
    # Embed the question
    question_embedding = emb.encode(texts= [req.question], task_type="RETRIEVAL_QUERY")[0]

    # Call hybrid_search SQL function
    doc_ids = req.document_ids or []
    db_results = sb.rpc(
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

    generator = RAGGenerator(model_name="mistral") # or llama3, etc.
    final_output = generator.generate_answer(req.question, db_results)

    return final_output

#     # 3. RETURN
#     return {
#         "answer": final_output["answer"],
#         "citations": final_output["sources"],
#         "debug_results": db_results.data # Keep this for now for debugging
#     }

#     # 3. RETURN
#     return {
#         "answer": final_output["answer"],
#         "citations": final_output["sources"],
#         "debug_results": db_results.data # Keep this for now for debugging
#     }

    # print(f"the results are {results}")

    # 3. Return structured response
    # return {"results": results.data}
