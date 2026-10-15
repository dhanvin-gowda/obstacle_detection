-- ==============================================================================
-- SUPABASE PGVECTOR SETUP FOR EMPLOYEES
-- Run this in your Supabase Dashboard: SQL Editor -> New Query -> Run
-- ==============================================================================

-- 1. Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- 2. Create the employees table
create table if not exists employees (
  id text primary key,
  name text not null,
  email text not null,
  skills text[] not null,
  experience_years integer not null,
  current_work text not null,
  content text not null,
  embedding vector(768),
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 3. Create an HNSW index on the vector embedding column for fast cosine similarity search
create index if not exists employees_embedding_hnsw_idx
on employees using hnsw (embedding vector_cosine_ops);

-- 4. Enable Row Level Security (RLS)
alter table employees enable row level security;

-- 5. Create policies to allow read and insert/update
-- (Adjust these if you use strict auth or service role keys)
create policy "Allow public read access to employees"
  on employees for select
  using (true);

create policy "Allow public insert and upsert to employees"
  on employees for insert
  with check (true);

create policy "Allow public update to employees"
  on employees for update
  using (true);

-- 6. Create RPC function for semantic vector search
create or replace function match_employees (
  query_embedding vector(768),
  match_threshold float default 0.2,
  match_count int default 5
)
returns table (
  id text,
  name text,
  email text,
  skills text[],
  experience_years int,
  current_work text,
  similarity float
)
language sql stable
as $$
  select
    employees.id,
    employees.name,
    employees.email,
    employees.skills,
    employees.experience_years,
    employees.current_work,
    1 - (employees.embedding <=> query_embedding) as similarity
  from employees
  where 1 - (employees.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
$$;
