-- Overthink This — Database Schema
-- Run in Supabase SQL editor before first launch.

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT UNIQUE,
    name TEXT NOT NULL,
    picture TEXT,
    is_guest BOOLEAN DEFAULT FALSE,
    tone_preference TEXT DEFAULT 'balanced',
    default_category TEXT DEFAULT 'other',
    plan_tier TEXT DEFAULT 'free',
    spirals_used_today INTEGER DEFAULT 0,
    spirals_used_date TEXT,
    spirals_total INTEGER DEFAULT 0,
    streak_count INTEGER DEFAULT 0,
    last_active TEXT,
    created_at TEXT NOT NULL,
    xp INTEGER DEFAULT 0,
    level INTEGER DEFAULT 1,
    phone_number TEXT,
    phone_verified BOOLEAN DEFAULT FALSE,
    ip_address TEXT,
    customization JSONB,
    unlocked_items JSONB DEFAULT '[]',
    deleted_count INTEGER DEFAULT 0,
    shared_count INTEGER DEFAULT 0,
    plan_expires_at TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

CREATE TABLE IF NOT EXISTS folders (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_folders_user_id ON folders(user_id);

CREATE TABLE IF NOT EXISTS spirals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    situation_text TEXT NOT NULL,
    category TEXT NOT NULL,
    tags JSONB DEFAULT '[]',
    tone_used TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'processing',
    resolved BOOLEAN DEFAULT FALSE,
    resolution_status TEXT,
    resolution_note TEXT,
    resolved_at TEXT,
    share_count INTEGER DEFAULT 0,
    outcomes JSONB DEFAULT '[]',
    verdict JSONB,
    error_message TEXT,
    flagged BOOLEAN DEFAULT FALSE,
    folder_id TEXT REFERENCES folders(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spirals_user_id ON spirals(user_id);
CREATE INDEX IF NOT EXISTS idx_spirals_created_at ON spirals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_spirals_folder_id ON spirals(folder_id);
CREATE INDEX IF NOT EXISTS idx_spirals_flagged ON spirals(flagged) WHERE flagged = TRUE;

CREATE TABLE IF NOT EXISTS phone_otps (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    phone_number TEXT NOT NULL,
    otp TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_tasks (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    period TEXT NOT NULL,
    progress INTEGER DEFAULT 0,
    target INTEGER NOT NULL,
    claimed BOOLEAN DEFAULT FALSE,
    claimed_at TEXT,
    created_at TEXT NOT NULL,
    tones_used JSONB DEFAULT '[]',
    categories_used JSONB DEFAULT '[]',
    UNIQUE (user_id, task_id, period)
);

CREATE TABLE IF NOT EXISTS payment_transactions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    package_id TEXT NOT NULL,
    amount FLOAT NOT NULL,
    currency TEXT NOT NULL DEFAULT 'usd',
    metadata JSONB,
    payment_status TEXT DEFAULT 'initiated',
    status TEXT DEFAULT 'open',
    plan_applied BOOLEAN DEFAULT FALSE,
    applied_at TEXT,
    amount_total INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT
);
