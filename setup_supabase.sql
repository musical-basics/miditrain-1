-- Run this in your Supabase SQL Editor manually

CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name TEXT NOT NULL,
    model_architecture TEXT NOT NULL,
    epochs INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    learning_rate NUMERIC NOT NULL,
    final_loss NUMERIC NOT NULL,
    weights_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Note: Because we are using the Service Role Key for our python script (server-side action), 
-- you do not strictly need Row Level Security (RLS) policies for INSERTs from the python script. 
-- The script will bypass RLS.
