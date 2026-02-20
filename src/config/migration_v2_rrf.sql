-- ============================================
-- Migration: Add structured metadata columns + RRF hybrid search
-- Run this in Supabase SQL Editor to upgrade the schema and search function
-- ============================================

-- Step 1: Add metadata columns to chunks table (idempotent)
ALTER TABLE public.chunks ADD COLUMN IF NOT EXISTS course_name TEXT;
ALTER TABLE public.chunks ADD COLUMN IF NOT EXISTS course_code TEXT;
ALTER TABLE public.chunks ADD COLUMN IF NOT EXISTS degree_name TEXT;
ALTER TABLE public.chunks ADD COLUMN IF NOT EXISTS section_type TEXT;

-- Step 2: Create indexes for metadata filtering
CREATE INDEX IF NOT EXISTS idx_chunks_course_name 
  ON public.chunks (course_name);

CREATE INDEX IF NOT EXISTS idx_chunks_course_code 
  ON public.chunks (course_code);

CREATE INDEX IF NOT EXISTS idx_chunks_degree_name 
  ON public.chunks (degree_name);

-- Step 3: Ensure full-text search indexes exist
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts 
  ON public.chunks 
  USING gin(to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm 
  ON public.chunks 
  USING gin(content gin_trgm_ops);

-- Step 4: Drop and recreate hybrid_search with RRF + metadata filters
DROP FUNCTION IF EXISTS public.hybrid_search;

CREATE OR REPLACE FUNCTION public.hybrid_search(
  p_question TEXT,
  p_question_embedding vector,
  p_document_ids UUID[],
  p_semantic_weight FLOAT DEFAULT 0.6,
  p_keyword_weight FLOAT DEFAULT 0.4,
  p_limit INT DEFAULT 20,
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
      AND 1 - (c.embedding <=> p_question_embedding) > 0.1
      AND (p_course_name IS NULL OR c.course_name ILIKE '%' || p_course_name || '%')
      AND (p_course_code IS NULL OR c.course_code ILIKE '%' || p_course_code || '%')
      AND (p_degree_name IS NULL OR c.degree_name ILIKE '%' || p_degree_name || '%')
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
      LEAST(1.0, ts_rank(to_tsvector('english', c.content), v_query) * 2.0)::FLOAT as keyword_score,
      ROW_NUMBER() OVER (
        ORDER BY ts_rank(to_tsvector('english', c.content), v_query) DESC
      ) as keyword_rank
    FROM public.chunks c
    JOIN public.documents d ON c.document_id = d.id
    WHERE (
      p_document_ids IS NULL 
      OR array_length(p_document_ids, 1) IS NULL 
      OR c.document_id = ANY(p_document_ids)
    )
      AND to_tsvector('english', c.content) @@ v_query
      AND (p_course_name IS NULL OR c.course_name ILIKE '%' || p_course_name || '%')
      AND (p_course_code IS NULL OR c.course_code ILIKE '%' || p_course_code || '%')
      AND (p_degree_name IS NULL OR c.degree_name ILIKE '%' || p_degree_name || '%')
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
      (COALESCE(p_semantic_weight / (60 + s.semantic_rank), 0.0) + 
       COALESCE(p_keyword_weight / (60 + k.keyword_rank), 0.0))::FLOAT as combined_score,
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
