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

-- Code to retrieve relevant text from postgres
create or replace function match_chunks (
  query_embedding vector(768),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  content text,
  source text, -- We will join this from the documents table
  page_number int,
  chunk_index int,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    chunks.id,
    chunks.content,
    documents.filename as source,
    chunks.page_number,
    chunks.chunk_index,
    1 - (chunks.embedding <=> query_embedding) as similarity
  from chunks
  join documents on chunks.document_id = documents.id
  where 1 - (chunks.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
end;
$$;

-- disable RLS
alter table documents enable row level security;
alter table chunks enable row level security;


-- For 384:
alter table chunks 
alter column embedding type vector(384);

-- 2. IMPORTANT: You MUST also update your RPC function
-- The function arguments must match the new table dimensions
create or replace function match_chunks (
  query_embedding vector(384), -- Updated from 768
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  content text,
  source text,
  page_number int,
  chunk_index int,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    chunks.id,
    chunks.content,
    documents.filename as source,
    chunks.page_number,
    chunks.chunk_index,
    1 - (chunks.embedding <=> query_embedding) as similarity
  from chunks
  join documents on chunks.document_id = documents.id
  where 1 - (chunks.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
end;
$$;
