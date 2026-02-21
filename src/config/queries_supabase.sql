create extension if not exists vector;

create table documents (
  id uuid primary key default gen_random_uuid(),
  filename text unique not null,
  title text,
  checksum text not null,
  program_url text,    -- canonical URL of the program's study guide page
  created_at timestamp with time zone default now()
);

create table chunks(
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  content text not null,
  embedding vector(768),
  page_number int,
  chunk_index int,
  metadata jsonb,  -- stores only { "source": "<filename>" }
  -- Program-level metadata (parsed from filename convention)
  doc_type text,       -- 'ter' | 'curriculum'
  program_type text,   -- 'masters' | 'bachelors' | 'premaster'
  program_name text,   -- e.g. 'artificial_intelligence', 'data_science'
  created_at timestamp with time zone default now()
);

-- CODE FOR HYBRID SEARCH

-- Migration: Add Hybrid Search (Semantic + BM25 Keyword) Support
-- Purpose: Combine vector similarity + keyword matching for better RAG accuracy
-- Especially effective for technical documents, PDFs, financial data

-- ============================================
-- Enable pg_trgm extension (for text similarity)
-- Trigram similarity, fuzzy matching, partial matches, typo tolerance
-- ============================================

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================
-- Create full-text search index on chunks
-- ============================================

-- Create Generalized Inverted Index for full-text search on content column of chunks table(ts-> text search )
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts 
  ON public.chunks 
  USING gin(to_tsvector('english', content));

-- Create trigram index for fuzzy matching (typos, partial matches)
CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm 
  ON public.chunks 
  USING gin(content gin_trgm_ops);

-- Indexes for program-level filtering
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type
  ON public.chunks (doc_type);

CREATE INDEX IF NOT EXISTS idx_chunks_program_type
  ON public.chunks (program_type);

CREATE INDEX IF NOT EXISTS idx_chunks_program_name
  ON public.chunks (program_name);

-- HNSW index for fast approximate nearest-neighbour vector search
-- Uses cosine distance (<=>).  m=16, ef_construction=64 are good defaults
-- for datasets < 10k rows.  This avoids brute-force scan.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
  ON public.chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ============================================
-- Hybrid Search with RRF (Reciprocal Rank Fusion) + Metadata Filtering
-- ============================================

CREATE OR REPLACE FUNCTION public.hybrid_search(
  p_question TEXT,
  p_question_embedding vector,
  p_document_ids UUID[],
  p_semantic_weight FLOAT DEFAULT 0.6,
  p_keyword_weight FLOAT DEFAULT 0.4,
  p_limit INT DEFAULT 10,
  -- Program-level filters (pass NULL to search across all programs)
  p_program_names TEXT[] DEFAULT NULL,  -- e.g. ARRAY['artificial_intelligence', 'data_science']
  p_doc_type TEXT DEFAULT NULL           -- e.g. 'ter' or 'curriculum'
)
RETURNS TABLE (
  id UUID,
  content TEXT,
  document_id UUID,
  filename TEXT,
  page_number INT,
  semantic_score FLOAT,
  keyword_score FLOAT,
  combined_score FLOAT,
  search_type TEXT,
  chunk_doc_type TEXT,
  chunk_program_type TEXT,
  chunk_program_name TEXT,
  program_url TEXT
) 
LANGUAGE plpgsql
AS $$
DECLARE
  v_query tsquery := plainto_tsquery('english', p_question);
  -- Inner candidate pool: fetch enough candidates for good fusion quality.
  -- With 100+ PDFs the corpus can exceed 10k chunks; even after program filtering
  -- there may be hundreds of candidates per leg. Use 5× the final limit,
  -- minimum 50, maximum 200.
  v_inner_limit INT := LEAST(GREATEST(p_limit * 5, 50), 200);
BEGIN
  RETURN QUERY
  WITH semantic_ranked AS (
    SELECT 
      c.id,
      c.content,
      c.document_id,
      d.filename,
      d.program_url,
      c.page_number,
      c.doc_type,
      c.program_type,
      c.program_name,
      (1 - (c.embedding <=> p_question_embedding))::FLOAT as semantic_score,
      ROW_NUMBER() OVER (ORDER BY c.embedding <=> p_question_embedding ASC) as semantic_rank
    FROM public.chunks c
    JOIN public.documents d ON c.document_id = d.id
    WHERE (
      p_document_ids IS NULL
      OR array_length(p_document_ids, 1) IS NULL
      OR c.document_id = ANY(p_document_ids)
    )
      AND c.embedding IS NOT NULL
      -- Program-level filters
      AND (p_program_names IS NULL OR array_length(p_program_names, 1) IS NULL OR c.program_name = ANY(p_program_names))
      AND (p_doc_type IS NULL OR c.doc_type = p_doc_type)
    ORDER BY c.embedding <=> p_question_embedding ASC
    LIMIT v_inner_limit
  ),
  keyword_ranked AS (
    SELECT 
      c.id,
      c.content,
      c.document_id,
      d.filename,
      d.program_url,
      c.page_number,
      c.doc_type,
      c.program_type,
      c.program_name,
      ts_rank_cd(to_tsvector('english', c.content), v_query)::FLOAT as keyword_score,
      ROW_NUMBER() OVER (
        ORDER BY ts_rank_cd(to_tsvector('english', c.content), v_query) DESC
      ) as keyword_rank
    FROM public.chunks c
    JOIN public.documents d ON c.document_id = d.id
    WHERE (
      p_document_ids IS NULL 
      OR array_length(p_document_ids, 1) IS NULL 
      OR c.document_id = ANY(p_document_ids)
    )
      AND to_tsvector('english', c.content) @@ v_query
      -- Program-level filters
      AND (p_program_names IS NULL OR array_length(p_program_names, 1) IS NULL OR c.program_name = ANY(p_program_names))
      AND (p_doc_type IS NULL OR c.doc_type = p_doc_type)
    ORDER BY ts_rank_cd(to_tsvector('english', c.content), v_query) DESC
    LIMIT v_inner_limit
  ),
  combined_results AS (
    SELECT 
      COALESCE(s.id, k.id) as id,
      COALESCE(s.content, k.content) as content,
      COALESCE(s.document_id, k.document_id) as document_id,
      COALESCE(s.filename, k.filename) as filename,
      COALESCE(s.page_number, k.page_number) as page_number,
      COALESCE(s.semantic_score, 0.0)::FLOAT as semantic_score,
      COALESCE(k.keyword_score, 0.0)::FLOAT as keyword_score,
      -- Standard RRF with k=60 (Cormack et al., 2009).
      -- k=60 gives smooth rank-based fusion across corpora of any size;
      -- rank 1 → 1/61≈0.016, rank 10 → 1/70≈0.014, rank 60 → 1/120≈0.008.
      -- Weighted by caller-supplied semantic/keyword weights (default 0.6/0.4).
      (COALESCE(p_semantic_weight / (60.0 + s.semantic_rank), 0.0) + 
       COALESCE(p_keyword_weight / (60.0 + k.keyword_rank), 0.0))::FLOAT as combined_score,
      CASE 
        WHEN s.id IS NOT NULL AND k.id IS NOT NULL THEN 'hybrid'
        WHEN s.id IS NOT NULL THEN 'semantic'
        ELSE 'keyword'
      END as search_type,
      COALESCE(s.doc_type, k.doc_type) as chunk_doc_type,
      COALESCE(s.program_type, k.program_type) as chunk_program_type,
      COALESCE(s.program_name, k.program_name) as chunk_program_name,
      COALESCE(s.program_url, k.program_url) as program_url
    FROM semantic_ranked s
    FULL OUTER JOIN keyword_ranked k ON s.id = k.id
  )
  SELECT *
  FROM combined_results
  ORDER BY combined_score DESC
  LIMIT p_limit;
END;
$$;