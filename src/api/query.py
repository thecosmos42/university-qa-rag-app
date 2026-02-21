from fastapi import APIRouter
from pydantic import BaseModel
import supabase
import time
from src.config.db_config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.ingestion.embedder_factory import get_embedder
from src.llm.llm_wrapper import LLMWrapper
from src.llm.query_rewriter import QueryRewriter
from src.llm.program_extractor import ProgramExtractor
from src.config.logger import logger
import re

router = APIRouter()
emb = get_embedder()
rewriter = QueryRewriter()
generator = LLMWrapper()
program_extractor = ProgramExtractor()
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

    search_query = req.question
    logger.info(f"Query: {search_query}")

    # Extract program-level filters (which programs + doc type this query targets)
    prog_filter = program_extractor.extract(search_query)
    logger.info(
        f"[ProgramFilter] programs={prog_filter.program_names or 'all'}"
    )
    # Convert to None when empty so SQL treats it as "no filter"
    p_program_names = prog_filter.program_names if prog_filter.program_names else None

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
            # Program-level filters derived from the query
            "p_program_names": p_program_names,
            "p_doc_type": None,
        }
    ).execute()
    t_search = time.time()

    for i, res in enumerate(db_results.data):
        logger.info(
            f"RANK {i+1} | {res.get('search_type')} | "
            f"score={res.get('combined_score'):.4f} | "
            f"program={res.get('chunk_program_name', '')} | "
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
                # Retry without program filters to broaden scope
                "p_program_names": None,
                "p_doc_type": None,
            }
        ).execute()
        for i, res in enumerate(db_results.data):
            logger.info(f"FALLBACK RANK {i+1} | Source: {res['filename']} Page: {res['page_number']}")
            logger.info(f"  - Combined (RRF) Score: {res.get('combined_score')}")
            logger.info(f"  - Search Type: {res.get('search_type')}")

    # Generate answer
    llm_result = generator.generate_answer(req.question, db_results)
    final_output = llm_result["text"]
    token_usage = llm_result["tokens"]
    t_llm = time.time()
    
    # Timing summary
    logger.info(
        f"[Timing] "
        f"embed={t_embed-t_start:.2f}s | "
        f"search={t_search-t_embed:.2f}s | "
        f"llm={t_llm-t_search:.2f}s | "
        f"total={t_llm-t_start:.2f}s"
    )
    # Token usage summary
    logger.info(
        f"[Tokens] "
        f"prompt={token_usage['prompt']} | "
        f"response={token_usage['response']} | "
        f"total={token_usage['total']}"
    )
    
    # Extract which SOURCE numbers the LLM actually cited in its answer
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
                    "programName": res.get("chunk_program_name", ""),
                    "programType": res.get("chunk_program_type", ""),
                    "docType": res.get("chunk_doc_type", ""),
                    "href": res.get("program_url") or None,
                })

    return {
        "response": final_output,
        "sources": sources,
        "debug": {
            "result_count": len(db_results.data),
            "tokens": token_usage,
            "program_filter": {
                "programs": prog_filter.program_names or "all",
            },
        }
    }