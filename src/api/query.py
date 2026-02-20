from fastapi import APIRouter
from pydantic import BaseModel
import supabase
import time
from src.config.db_config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.ingestion.embedder import Embedder
from src.api.RAGGenerator import RAGGenerator
from src.rag.query_rewriter import QueryRewriter
from src.config.logger import logger

router = APIRouter()
emb = Embedder()
rewriter = QueryRewriter()
generator = RAGGenerator()  # Reuse across requests

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
    t_start = time.time()

    # # Query rewriting disabled for now
    # rewritten = rewriter.rewrite(req.question)
    # t_rewrite = time.time()
    # logger.info(f"Rewritten query: {rewritten.rewritten_query}")

    # Use original question directly
    search_query = req.question
    logger.info(f"Query: {search_query}")

    # Embed the question for semantic matching
    question_embedding = emb.encode(
        texts=[search_query], 
        task_type="RETRIEVAL_QUERY"
    )[0]
    t_embed = time.time()

    # Call hybrid_search with RRF + metadata filters
    doc_ids = req.document_ids or []
    db_results = sb.rpc(
        "hybrid_search",
        {
            "p_question": search_query,
            "p_question_embedding": question_embedding,
            "p_document_ids": doc_ids,
            "p_semantic_weight": req.semantic_weight,
            "p_keyword_weight": req.keyword_weight,
            "p_limit": req.limit,
            # No metadata filters without query rewriter
            "p_course_name": None,
            "p_course_code": None,
            "p_degree_name": None,
        }
    ).execute()
    t_search = time.time()

    # Compact logging — one line per result instead of 6
    for i, res in enumerate(db_results.data):
        logger.info(
            f"RANK {i+1} | {res.get('search_type')} | "
            f"score={res.get('combined_score'):.4f} | "
            f"{res.get('chunk_course_name', '')} | "
            f"{res['content'][:80]}..."
        )

    if not db_results.data:
        logger.warning("DB returned ZERO results for this query.")
        
        # Fallback: retry with broader search if initial returned nothing
        logger.info("Retrying with broader search...")
        db_results = sb.rpc(
            "hybrid_search",
            {
                "p_question": search_query,
                "p_question_embedding": question_embedding,
                "p_document_ids": doc_ids,
                "p_semantic_weight": req.semantic_weight,
                "p_keyword_weight": req.keyword_weight,
                "p_limit": req.limit,
                "p_course_name": None,
                "p_course_code": None,
                "p_degree_name": None,
            }
        ).execute()
        for i, res in enumerate(db_results.data):
            logger.info(f"FALLBACK RANK {i+1} | Source: {res['filename']} Page: {res['page_number']}")
            logger.info(f"  - Combined (RRF) Score: {res.get('combined_score')}")
            logger.info(f"  - Search Type: {res.get('search_type')}")

    # Step 5: Generate answer
    final_output = generator.generate_answer(req.question, db_results)
    t_llm = time.time()
    
    # Timing summary
    logger.info(
        f"[Timing] "
        f"embed={t_embed-t_start:.2f}s | "
        f"search={t_search-t_embed:.2f}s | "
        f"llm={t_llm-t_search:.2f}s | "
        f"total={t_llm-t_start:.2f}s"
    )
    
    # Extract which SOURCE numbers the LLM actually cited in its answer
    import re
    cited_indices = set(int(n) for n in re.findall(r'\[SOURCE\s+(\d+)', final_output))

    # Only return sources that were actually cited
    sources = []
    if db_results.data:
        for i, res in enumerate(db_results.data):
            idx = i + 1
            if idx in cited_indices:
                sources.append({
                    "index": idx,
                    "label": res["filename"],
                    "page": res.get("page_number"),
                    "courseName": res.get("chunk_course_name", ""),
                })

    return {
        "response": final_output,
        "sources": sources,
        "debug": {
            "result_count": len(db_results.data),
        }
    }