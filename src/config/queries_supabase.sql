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

CREATE OR REPLACE FUNCTION public.hybrid_search(
  p_question TEXT,
  p_question_embedding vector,
  p_document_ids UUID[],
  p_semantic_weight FLOAT DEFAULT 0.6,
  p_keyword_weight FLOAT DEFAULT 0.4,
  p_limit INT DEFAULT 20
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
  search_type TEXT
) 
LANGUAGE plpgsql
AS $$
DECLARE
  v_query tsquery := plainto_tsquery('english', p_question);
BEGIN
  RETURN QUERY
  WITH semantic_results AS (
    SELECT 
      c.id,
      c.content,
      c.document_id,
      d.filename,
      c.page_number,
      -- Use 1 - distance to get similarity
      (1 - (c.embedding <=> p_question_embedding))::FLOAT as semantic_score,
      0.0::FLOAT as keyword_score,
      ((1 - (c.embedding <=> p_question_embedding)) * p_semantic_weight)::FLOAT as combined_score,
      'semantic'::TEXT as search_type
    FROM public.chunks c
    JOIN public.documents d ON c.document_id = d.id
    WHERE c.document_id = ANY(p_document_ids)
      AND c.embedding IS NOT NULL
      -- Minimum threshold to filter out noise
      AND 1 - (c.embedding <=> p_question_embedding) > 0.2
  ),
  keyword_results AS (
    SELECT 
      c.id,
      c.content,
      c.document_id,
      d.filename,
      c.page_number,
      0.0::FLOAT as semantic_score,
      -- ts_rank calculates how well the keywords match the content
      LEAST(1.0, ts_rank(to_tsvector('english', c.content), v_query) * 2.0)::FLOAT as keyword_score,
      (LEAST(1.0, ts_rank(to_tsvector('english', c.content), v_query) * 2.0) * p_keyword_weight)::FLOAT as combined_score,
      'keyword'::TEXT as search_type
    FROM public.chunks c
    JOIN public.documents d ON c.document_id = d.id
    WHERE (
      -- If the array is empty or null, ignore this filter
      p_document_ids IS NULL 
      OR array_length(p_document_ids, 1) IS NULL 
      OR c.document_id = ANY(p_document_ids)
    )
      -- The @@ operator checks for a match between the query and the text
      AND to_tsvector('english', c.content) @@ v_query
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
      (COALESCE(s.semantic_score, 0.0) * p_semantic_weight + 
       COALESCE(k.keyword_score, 0.0) * p_keyword_weight)::FLOAT as combined_score,
      CASE 
        WHEN s.id IS NOT NULL AND k.id IS NOT NULL THEN 'hybrid'
        WHEN s.id IS NOT NULL THEN 'semantic'
        ELSE 'keyword'
      END as search_type
    FROM semantic_results s
    FULL OUTER JOIN keyword_results k ON s.id = k.id
  )
  SELECT *
  FROM combined_results
  ORDER BY combined_score DESC
  LIMIT p_limit;
END;
$$;