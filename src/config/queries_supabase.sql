create extension if not exists vector;

create table documents (
  id uuid primary key default gen_random_uuid(),
  filename text unique not null,
  title text,
  checksum text not null,
  created_at timestamp with time zone default now()
);

create table chunks(
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  content text not null,
  embedding vector(768),
  page_number int,
  chunk_index int,
  metadata jsonb,
  -- Structured metadata for filtering and boosting
  course_name text,
  course_code text,
  degree_name text,
  section_type text,  -- 'course_detail', 'program_overview', 'course_list', etc.
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

-- Index for metadata-based filtering
CREATE INDEX IF NOT EXISTS idx_chunks_course_name 
  ON public.chunks (course_name);

CREATE INDEX IF NOT EXISTS idx_chunks_course_code 
  ON public.chunks (course_code);

CREATE INDEX IF NOT EXISTS idx_chunks_degree_name 
  ON public.chunks (degree_name);

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
  p_course_name TEXT DEFAULT NULL,
  p_course_code TEXT DEFAULT NULL,
  p_degree_name TEXT DEFAULT NULL
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
  chunk_course_name TEXT,
  chunk_course_code TEXT
) 
LANGUAGE plpgsql
AS $$
DECLARE
  v_query tsquery := plainto_tsquery('english', p_question);
  -- Inner limit: fetch more candidates than final limit for better fusion
  v_inner_limit INT := LEAST(p_limit * 3, 30);
BEGIN
  RETURN QUERY
  WITH semantic_ranked AS (
    SELECT 
      c.id,
      c.content,
      c.document_id,
      d.filename,
      c.page_number,
      c.course_name,
      c.course_code,
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
      -- Metadata filters (only applied if provided)
      AND (p_course_name IS NULL OR c.course_name ILIKE '%' || p_course_name || '%')
      AND (p_course_code IS NULL OR c.course_code ILIKE '%' || p_course_code || '%')
      AND (p_degree_name IS NULL OR c.degree_name ILIKE '%' || p_degree_name || '%')
    ORDER BY c.embedding <=> p_question_embedding ASC
    LIMIT v_inner_limit
  ),
  keyword_ranked AS (
    SELECT 
      c.id,
      c.content,
      c.document_id,
      d.filename,
      c.page_number,
      c.course_name,
      c.course_code,
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
      -- Metadata filters
      AND (p_course_name IS NULL OR c.course_name ILIKE '%' || p_course_name || '%')
      AND (p_course_code IS NULL OR c.course_code ILIKE '%' || p_course_code || '%')
      AND (p_degree_name IS NULL OR c.degree_name ILIKE '%' || p_degree_name || '%')
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
      -- RRF with k=1 for small corpus (~120 chunks)
      -- k=60 (standard) gives near-identical scores; k=1 gives strong separation
      -- Rank 1: 1/2=0.50, Rank 2: 1/3=0.33, Rank 10: 1/11=0.09
      (COALESCE(p_semantic_weight / (1 + s.semantic_rank), 0.0) + 
       COALESCE(p_keyword_weight / (1 + k.keyword_rank), 0.0))::FLOAT as combined_score,
      CASE 
        WHEN s.id IS NOT NULL AND k.id IS NOT NULL THEN 'hybrid'
        WHEN s.id IS NOT NULL THEN 'semantic'
        ELSE 'keyword'
      END as search_type,
      COALESCE(s.course_name, k.course_name) as chunk_course_name,
      COALESCE(s.course_code, k.course_code) as chunk_course_code
    FROM semantic_ranked s
    FULL OUTER JOIN keyword_ranked k ON s.id = k.id
  )
  SELECT *
  FROM combined_results
  ORDER BY combined_score DESC
  LIMIT p_limit;
END;
$$;