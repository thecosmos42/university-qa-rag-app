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


-- disable RLS
alter table documents enable row level security;
alter table chunks enable row level security;
