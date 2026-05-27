"""
Overthink This — FastAPI backend.

Run:  uvicorn server:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import json
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# HuggingFace Inference Router (OpenAI-compatible). Used as the
# secondary AI lane — when every Gemini model in the rotation refuses
# or times out we try DeepSeek (and a couple known-good HF chat models)
# before falling back to the offline placeholder payload.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", "")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
CORS_ORIGINS_EXTRA = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]

SESSION_TTL_DAYS = 30
FREE_LIFETIME_LIMIT = 3
MAX_GUESTS_PER_IP_PER_DAY = 5
OTP_TTL_MINUTES = 10

# ---------------------------------------------------------------------------
# DB pool lifecycle
# ---------------------------------------------------------------------------

pool: Optional[asyncpg.Pool] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pool
    if DATABASE_URL:
        # Supabase direct connections require SSL. asyncpg ignores ?sslmode= URL
        # params, so pass ssl explicitly when the URL looks like Supabase or
        # when sslmode=require is in the URL.
        url = DATABASE_URL
        needs_ssl = "supabase.co" in url or "supabase.com" in url or "sslmode=require" in url
        if "?" in url:
            url = url.split("?", 1)[0]
        pool = await asyncpg.create_pool(
            url,
            min_size=1,
            max_size=10,
            statement_cache_size=0,  # pooler-friendly
            ssl="require" if needs_ssl else None,
        )
        # Idempotent migrations. EACH statement runs in its own try so
        # one failure doesn't cascade and skip the rest of the block —
        # that bug previously meant text_checks/compatibility_tests
        # never got created when an earlier ALTER hit anything funny.
        migrations: List[tuple[str, str]] = [
            ("users.stripe_customer_id",        "ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT"),
            ("users.stripe_subscription_id",    "ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_subscription_id TEXT"),
            # Onboarding step 2 — bio + personality + frequency self-report.
            ("users.bio",                       "ALTER TABLE users ADD COLUMN IF NOT EXISTS bio TEXT"),
            ("users.personality",               "ALTER TABLE users ADD COLUMN IF NOT EXISTS personality TEXT"),
            ("users.spiral_frequency",          "ALTER TABLE users ADD COLUMN IF NOT EXISTS spiral_frequency TEXT"),
            # First-time pricing gate.
            ("users.has_ever_subscribed",       "ALTER TABLE users ADD COLUMN IF NOT EXISTS has_ever_subscribed BOOLEAN DEFAULT FALSE"),
            # Feature #10 — Spiral Soundtrack JSONB sidecar.
            ("spirals.soundtrack",              "ALTER TABLE spirals ADD COLUMN IF NOT EXISTS soundtrack JSONB"),
            # Feature #9 — Streak with stakes + Pro freezes.
            ("users.last_spiral_date",          "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_spiral_date TEXT"),
            ("users.streak_freezes_remaining",  "ALTER TABLE users ADD COLUMN IF NOT EXISTS streak_freezes_remaining INTEGER DEFAULT 3"),
            ("users.streak_freezes_month",      "ALTER TABLE users ADD COLUMN IF NOT EXISTS streak_freezes_month TEXT"),
            # Multi-item archive — text-check + compatibility persistence.
            # Each row mirrors spiral's folder/accent/flagged so the
            # frontend can treat all three item kinds uniformly.
            ("text_checks",
                """CREATE TABLE IF NOT EXISTS text_checks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT,
                    draft TEXT NOT NULL,
                    context TEXT,
                    relationship TEXT,
                    result JSONB NOT NULL,
                    folder_id TEXT,
                    accent_color TEXT,
                    flagged BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )"""),
            ("compatibility_tests",
                """CREATE TABLE IF NOT EXISTS compatibility_tests (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT,
                    person_a JSONB NOT NULL,
                    person_b JSONB NOT NULL,
                    result JSONB NOT NULL,
                    folder_id TEXT,
                    accent_color TEXT,
                    flagged BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )"""),
            # Cross-type pair links.
            ("item_pairs",
                """CREATE TABLE IF NOT EXISTS item_pairs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    a_type TEXT NOT NULL,
                    a_id TEXT NOT NULL,
                    b_type TEXT NOT NULL,
                    b_id TEXT NOT NULL,
                    note TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )"""),
            ("idx_item_pairs_a",
                "CREATE INDEX IF NOT EXISTS idx_item_pairs_a ON item_pairs (user_id, a_type, a_id)"),
            ("idx_item_pairs_b",
                "CREATE INDEX IF NOT EXISTS idx_item_pairs_b ON item_pairs (user_id, b_type, b_id)"),
            # Streak activity grid (heatmap) — one row per user per
            # date. Records active days and freeze-rescued days.
            ("streak_activity",
                """CREATE TABLE IF NOT EXISTS streak_activity (
                    user_id TEXT NOT NULL,
                    activity_date DATE NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'active',
                    spiral_count INTEGER NOT NULL DEFAULT 0,
                    text_check_count INTEGER NOT NULL DEFAULT 0,
                    compat_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, activity_date)
                )"""),
            ("idx_streak_activity_user_date",
                "CREATE INDEX IF NOT EXISTS idx_streak_activity_user_date ON streak_activity (user_id, activity_date DESC)"),
        ]
        try:
            async with pool.acquire() as conn:
                for name, sql in migrations:
                    try:
                        await conn.execute(sql)
                        print(f"[lifespan] migration ✅ {name}")
                    except Exception as inner:
                        # Don't let one failure poison the rest. Log it
                        # loudly so the next deploy surfaces it.
                        print(f"[lifespan] migration ❌ {name}: {type(inner).__name__}: {inner}")
        except Exception as exc:
            print(f"[lifespan] migration block error: {type(exc).__name__}: {exc}")
    yield
    if pool:
        await pool.close()


# Disable the public Swagger UI / ReDoc / openapi.json pages so the API surface
# isn't browsable by anyone who finds the URL.
app = FastAPI(
    title="Overthink This API",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

DEFAULT_CORS = [
    "http://localhost:8081",
    "http://localhost:8082",
    "http://localhost:19006",
    "http://localhost:3000",
    "exp://localhost:8081",
    "exp://localhost:8082",
]

# Allow any localhost/127.0.0.1 port during dev — Expo picks 8081, 8082, or
# whatever's free, and we don't want to chase the port number every restart.
# In production you'd swap this for a fixed origin list.
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_CORS + CORS_ORIGINS_EXTRA,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-App-Key", "Accept"],
)

# ---------------------------------------------------------------------------
# App-level API key gate
#
# Every request (except Stripe's webhook, which authenticates itself via the
# signature header) must include X-App-Key matching INTERNAL_API_KEY. This
# isn't "real" auth — that's still done with session tokens — but it stops
# random people on the internet from probing the endpoints.
# ---------------------------------------------------------------------------

INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")

@app.middleware("http")
async def app_key_gate(request: Request, call_next):
    path = request.url.path
    # Stripe webhook authenticates itself with the signature header, so it
    # gets a pass. Health check too.
    if (
        path.startswith("/api/payments/webhook")
        or path.startswith("/api/payments/return")  # HTTPS bridge for Stripe redirect
        or path.startswith("/api/payments/diag")    # browser-visible Stripe diagnostic
        or path.startswith("/api/diag")
        or path == "/"
    ):
        return await call_next(request)
    if INTERNAL_API_KEY:
        provided = request.headers.get("x-app-key")
        if provided != INTERNAL_API_KEY:
            return JSONResponse({"detail": "Forbidden"}, status_code=403)
    return await call_next(request)


@app.get("/")
async def root():
    return {"ok": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_iso_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def db_required():
    if pool is None:
        raise HTTPException(503, "Database unavailable")


async def fetch_user(user_id: str) -> Optional[dict]:
    db_required()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user_id)
    return dict(row) if row else None


def user_public(u: dict) -> dict:
    # Hide nothing too sensitive — these fields drive the UI.
    customization = u.get("customization")
    if isinstance(customization, str):
        try:
            customization = json.loads(customization)
        except Exception:
            customization = None
    unlocked = u.get("unlocked_items") or []
    if isinstance(unlocked, str):
        try:
            unlocked = json.loads(unlocked)
        except Exception:
            unlocked = []
    # Streak view computed at read time — so a user who hasn't opened
    # the app for a week sees the broken streak immediately on next
    # login, without waiting for a background job. streak_count is
    # OVERRIDDEN with the effective (possibly zero) value so legacy
    # UI keeps rendering the right number.
    streak_view = _compute_streak_view(u) if "_compute_streak_view" in globals() else {
        "streak_count": u.get("streak_count") or 0,
        "freezes_remaining": 3,
        "freezes_max": 3,
        "is_pro": False,
        "days_since_last_spiral": None,
        "in_danger": False,
        "broken_today": False,
        "last_spiral_date": None,
    }
    return {
        "user_id": u["user_id"],
        "email": u.get("email"),
        "name": u["name"],
        "picture": u.get("picture"),
        "is_guest": bool(u.get("is_guest")),
        "tone_preference": u.get("tone_preference") or "balanced",
        "default_category": u.get("default_category") or "other",
        "plan_tier": u.get("plan_tier") or "free",
        "spirals_used_today": u.get("spirals_used_today") or 0,
        "spirals_total": u.get("spirals_total") or 0,
        # Effective streak — zero when the gap exceeds the freeze
        # rescue window. Raw column value would be misleading here.
        "streak_count": streak_view["streak_count"],
        "xp": u.get("xp") or 0,
        "level": u.get("level") or 1,
        "phone_number": u.get("phone_number"),
        "phone_verified": bool(u.get("phone_verified")),
        "customization": customization,
        "unlocked_items": unlocked,
        "bio": u.get("bio") or None,
        "personality": u.get("personality") or None,
        "spiral_frequency": u.get("spiral_frequency") or None,
        # Drives the "FIRST-TIME OFFER" strikethrough on the paywall —
        # any returning subscriber sees the real price with no discount.
        "has_ever_subscribed": bool(u.get("has_ever_subscribed")),
        # Streak-with-stakes (#9). Frontend reads these to render the
        # flame state, "in danger" badge, and freezes counter.
        "streak_freezes_remaining": streak_view["freezes_remaining"],
        "streak_freezes_max": streak_view["freezes_max"],
        "streak_in_danger": streak_view["in_danger"],
        "streak_broken_today": streak_view["broken_today"],
        "days_since_last_spiral": streak_view["days_since_last_spiral"],
        "created_at": u.get("created_at"),
    }


def spiral_public(s: dict) -> dict:
    outcomes = s.get("outcomes") or []
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            outcomes = []
    verdict = s.get("verdict")
    if isinstance(verdict, str):
        try:
            verdict = json.loads(verdict)
        except Exception:
            verdict = None
    tags = s.get("tags") or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []
    soundtrack = s.get("soundtrack")
    if isinstance(soundtrack, str):
        try:
            soundtrack = json.loads(soundtrack)
        except Exception:
            soundtrack = None
    return {
        "id": s["id"],
        "user_id": s["user_id"],
        "situation_text": s["situation_text"],
        "category": s["category"],
        "tags": tags,
        "tone_used": s["tone_used"],
        "status": s["status"],
        "resolved": bool(s.get("resolved")),
        "resolution_status": s.get("resolution_status"),
        "resolution_note": s.get("resolution_note"),
        "resolved_at": s.get("resolved_at"),
        "share_count": s.get("share_count") or 0,
        "outcomes": outcomes,
        "verdict": verdict,
        "soundtrack": soundtrack,
        "error_message": s.get("error_message"),
        "flagged": bool(s.get("flagged")),
        "folder_id": s.get("folder_id"),
        "accent_color": s.get("accent_color"),
        "name": s.get("name"),
        "created_at": s["created_at"],
    }


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def get_token(
    authorization: Optional[str] = Header(None),
    session_token: Optional[str] = Cookie(None),
) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return session_token


async def get_current_user(token: Optional[str] = Depends(get_token)) -> dict:
    if not token:
        raise HTTPException(401, "Not authenticated")
    db_required()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT u.* FROM sessions s JOIN users u ON u.user_id = s.user_id "
            "WHERE s.session_token = $1 AND s.expires_at > $2",
            token,
            now_iso(),
        )
    if not row:
        raise HTTPException(401, "Invalid session")
    return dict(row)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

async def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)).isoformat()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO sessions (session_token, user_id, expires_at, created_at) VALUES ($1,$2,$3,$4)",
            token, user_id, expires, now_iso(),
        )
    return token


def attach_session_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(
        "session_token",
        token,
        max_age=SESSION_TTL_DAYS * 24 * 3600,
        httponly=True,
        samesite="lax",
        secure=False,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GuestRequest(BaseModel):
    name: str = Field(default="Guest", max_length=40)


class GoogleRequest(BaseModel):
    access_token: str


class PreferencesRequest(BaseModel):
    tone_preference: Optional[str] = None
    default_category: Optional[str] = None
    # Onboarding step 2 — personality + self-reported spiral frequency.
    # bio is a free-form blurb the user writes about themselves.
    bio: Optional[str] = None
    personality: Optional[str] = None
    spiral_frequency: Optional[str] = None


class CustomizeRequest(BaseModel):
    active_title: Optional[str] = None
    name_color: Optional[str] = None
    card_theme: Optional[str] = None


class PhoneSendRequest(BaseModel):
    phone_number: str


class PhoneVerifyRequest(BaseModel):
    phone_number: str
    otp: str


class SpiralCreate(BaseModel):
    situation_text: str = Field(min_length=1, max_length=2000)
    category: str
    tone: str = "balanced"


class SpiralResolve(BaseModel):
    resolved: bool
    resolution_status: Optional[str] = None
    resolution_note: Optional[str] = None


class SpiralPatch(BaseModel):
    category: Optional[str] = None
    tags: Optional[list[str]] = None
    flagged: Optional[bool] = None
    folder_id: Optional[str] = None  # null = remove from folder
    accent_color: Optional[str] = None  # null clears
    name: Optional[str] = None  # rename spiral


class FolderCreate(BaseModel):
    name: str
    color: Optional[str] = None


class FolderRename(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None


class CheckoutRequest(BaseModel):
    package_id: str
    origin_url: str


# ---------------------------------------------------------------------------
# Auth — guest / google / me / logout
# ---------------------------------------------------------------------------

@app.post("/api/auth/guest")
async def auth_guest(body: GuestRequest, request: Request, response: Response):
    db_required()
    ip = request.client.host if request.client else "0.0.0.0"
    async with pool.acquire() as conn:
        # Rate limit guest creation per IP per 24h
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        guest_count = await conn.fetchval(
            "SELECT COUNT(*) FROM users WHERE is_guest = TRUE AND ip_address = $1 AND created_at > $2",
            ip, cutoff,
        )
        if guest_count and guest_count >= MAX_GUESTS_PER_IP_PER_DAY:
            raise HTTPException(429, "Too many guest accounts from this IP. Please sign in instead.")

        user_id = f"guest_{uuid.uuid4().hex[:16]}"
        await conn.execute(
            """
            INSERT INTO users (user_id, name, is_guest, plan_tier, created_at, ip_address, last_active, customization, unlocked_items)
            VALUES ($1,$2,TRUE,'free',$3,$4,$3,$5,$6)
            """,
            user_id, body.name or "Guest", now_iso(), ip,
            json.dumps({}), json.dumps([]),
        )
        u = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user_id)

    token = await create_session(user_id)
    attach_session_cookie(response, token)
    return {"user": user_public(dict(u)), "session_token": token}


@app.post("/api/auth/google")
async def auth_google(body: GoogleRequest, request: Request, response: Response):
    db_required()
    # Verify the Google access token and fetch profile
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {body.access_token}"},
            )
            r.raise_for_status()
            info = r.json()
        except Exception as exc:
            raise HTTPException(401, f"Google auth failed: {exc}")

    email = info.get("email")
    if not email:
        raise HTTPException(401, "Google profile missing email")

    name = info.get("name") or email.split("@")[0]
    picture = info.get("picture")
    ip = request.client.host if request.client else None

    # Idempotent upsert keyed on email. The previous SELECT-then-INSERT
    # pattern races: if two requests for the same Google account arrive
    # before the first commits, both see existing=None and both attempt
    # INSERT — the second one hits a unique-constraint violation and the
    # endpoint 500s. ON CONFLICT lets the second request quietly update
    # the existing row instead. The new user_id is only used if the row
    # didn't exist already; on conflict we read back whatever id is on
    # the row that's actually in the table.
    new_user_id = f"user_{uuid.uuid4().hex[:16]}"
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (user_id, email, name, picture, is_guest, plan_tier,
                               created_at, ip_address, last_active, customization, unlocked_items)
            VALUES ($1,$2,$3,$4,FALSE,'free',$5,$6,$5,$7,$8)
            ON CONFLICT (email) DO UPDATE
              SET name = EXCLUDED.name,
                  picture = EXCLUDED.picture,
                  last_active = EXCLUDED.last_active
            """,
            new_user_id, email, name, picture, now_iso(), ip,
            json.dumps({}), json.dumps([]),
        )
        u = await conn.fetchrow("SELECT * FROM users WHERE email = $1", email)
        user_id = u["user_id"] if u else new_user_id

    token = await create_session(user_id)
    attach_session_cookie(response, token)
    return {"user": user_public(dict(u)), "session_token": token}


@app.get("/api/auth/me")
async def auth_me(user: dict = Depends(get_current_user)):
    return {"user": user_public(user)}


# ---------------------------------------------------------------------------
# Streak activity grid — GitHub-style heatmap data
# ---------------------------------------------------------------------------
#
# Returns the most recent N days as a chronological list of
# {date, kind, total} entries. kind is one of:
#   • "active"   — user logged at least one item that day
#   • "freeze"   — gap day rescued by a Pro streak freeze
#   • "miss"     — no activity, no freeze (just empty)
# The frontend renders this as a 7×N/7 grid coloured by kind.
@app.get("/api/streak/activity")
async def get_streak_activity(days: int = 84, user: dict = Depends(get_current_user)):
    db_required()
    await _ensure_archive_tables()
    days = max(7, min(int(days or 84), 365))  # clamp to sane range
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)
    rows_by_date: Dict[str, dict] = {}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT activity_date::text AS d, kind,
                      spiral_count, text_check_count, compat_count
                 FROM streak_activity
                WHERE user_id = $1 AND activity_date >= $2::date
                ORDER BY activity_date ASC""",
            user["user_id"], start.isoformat(),
        )
        for r in rows:
            rows_by_date[r["d"]] = {
                "kind": r["kind"],
                "spiral": int(r["spiral_count"] or 0),
                "text_check": int(r["text_check_count"] or 0),
                "compat": int(r["compat_count"] or 0),
            }
    out = []
    for i in range(days):
        d = (start + timedelta(days=i)).isoformat()
        cell = rows_by_date.get(d)
        if cell is None:
            out.append({"date": d, "kind": "miss", "total": 0, "spiral": 0, "text_check": 0, "compat": 0})
        else:
            total = cell["spiral"] + cell["text_check"] + cell["compat"]
            out.append({
                "date": d,
                "kind": cell["kind"],
                "total": total,
                "spiral": cell["spiral"],
                "text_check": cell["text_check"],
                "compat": cell["compat"],
            })
    return {
        "days": out,
        "streak_count": int(user.get("streak_count") or 0),
        "freezes_remaining": int(user.get("streak_freezes_remaining") or 0),
        "is_pro": _is_pro_tier(user.get("plan_tier")),
    }


@app.post("/api/auth/logout")
async def auth_logout(token: Optional[str] = Depends(get_token)):
    if token and pool:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM sessions WHERE session_token = $1", token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("session_token")
    return resp


@app.patch("/api/auth/preferences")
async def update_preferences(body: PreferencesRequest, user: dict = Depends(get_current_user)):
    sets, args = [], []
    if body.tone_preference is not None:
        sets.append(f"tone_preference = ${len(args)+1}")
        args.append(body.tone_preference)
    if body.default_category is not None:
        sets.append(f"default_category = ${len(args)+1}")
        args.append(body.default_category)
    # Onboarding step 2 fields — accepted via the same endpoint so the
    # onboarding screen can write them in one PATCH after the user finishes.
    if body.bio is not None:
        sets.append(f"bio = ${len(args)+1}")
        args.append(body.bio[:500] if body.bio else None)  # safety cap
    if body.personality is not None:
        sets.append(f"personality = ${len(args)+1}")
        args.append(body.personality[:60] if body.personality else None)
    if body.spiral_frequency is not None:
        sets.append(f"spiral_frequency = ${len(args)+1}")
        args.append(body.spiral_frequency[:30] if body.spiral_frequency else None)
    if not sets:
        return {"user": user_public(user)}
    args.append(user["user_id"])
    async with pool.acquire() as conn:
        await conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE user_id = ${len(args)}", *args)
        u = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user["user_id"])
    return {"user": user_public(dict(u))}


@app.patch("/api/auth/customize")
async def update_customization(body: CustomizeRequest, user: dict = Depends(get_current_user)):
    if (user.get("plan_tier") or "free") == "free":
        raise HTTPException(402, "Pro plan required for customization")
    # Validate ownership of requested cosmetics
    unlocked = user.get("unlocked_items") or []
    if isinstance(unlocked, str):
        try:
            unlocked = json.loads(unlocked)
        except Exception:
            unlocked = []
    requested = {}
    if body.active_title is not None:
        if body.active_title and f"title:{body.active_title}" not in unlocked:
            raise HTTPException(403, "Title not unlocked")
        requested["active_title"] = body.active_title
    if body.name_color is not None:
        # Default colors (white #F4F3F7, black #161520) are ALWAYS allowed,
        # plus null which clears the choice. Everything else needs unlock.
        DEFAULT_COLORS = {"#F4F3F7", "#161520"}
        if (body.name_color
            and body.name_color not in DEFAULT_COLORS
            and f"name_color:{body.name_color}" not in unlocked):
            raise HTTPException(403, "Name color not unlocked")
        requested["name_color"] = body.name_color
    if body.card_theme is not None:
        if body.card_theme and body.card_theme != "dark" and f"card_theme:{body.card_theme}" not in unlocked:
            raise HTTPException(403, "Card theme not unlocked")
        requested["card_theme"] = body.card_theme

    current = user.get("customization") or {}
    if isinstance(current, str):
        try:
            current = json.loads(current)
        except Exception:
            current = {}
    current.update({k: v for k, v in requested.items() if v is not None})

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET customization = $1 WHERE user_id = $2",
            json.dumps(current), user["user_id"],
        )
        u = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user["user_id"])
    return {"user": user_public(dict(u))}


# ---------------------------------------------------------------------------
# Phone OTP (dev — no SMS provider, OTP logged to console)
# ---------------------------------------------------------------------------

async def _send_sms_via_twilio(to_number: str, body_text: str) -> tuple[bool, str]:
    """Returns (success, info). Doesn't raise — caller decides whether the
    OTP attempt should still 'succeed' if SMS dispatch failed."""
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        return False, "twilio_not_configured"
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                data={"From": TWILIO_FROM_NUMBER, "To": to_number, "Body": body_text},
            )
        if 200 <= resp.status_code < 300:
            return True, "sent"
        return False, f"twilio_{resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return False, f"twilio_exception: {type(exc).__name__}: {exc}"


@app.post("/api/auth/phone/send")
async def phone_send(body: PhoneSendRequest, user: dict = Depends(get_current_user)):
    import random
    otp = f"{random.randint(0, 999999):06d}"
    expires = (datetime.now(timezone.utc) + timedelta(minutes=OTP_TTL_MINUTES)).isoformat()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO phone_otps (user_id, phone_number, otp, expires_at, created_at) VALUES ($1,$2,$3,$4,$5)",
            user["user_id"], body.phone_number, otp, expires, now_iso(),
        )

    sent, info = await _send_sms_via_twilio(
        body.phone_number,
        f"Your Overthink This code is {otp}. Expires in {OTP_TTL_MINUTES} minutes.",
    )
    print(f"[OTP] {body.phone_number} -> {otp} ({'SMS sent' if sent else f'NO SMS ({info})'})")

    # When Twilio isn't configured (yet — no FROM_NUMBER), still return the
    # OTP in the response so dev/testing isn't blocked. Once a real
    # FROM_NUMBER is set in prod env vars, dev_otp won't be exposed.
    out: dict = {"ok": True, "message": "OTP sent" if sent else "OTP generated (no SMS)"}
    if not sent:
        out["dev_otp"] = otp
        out["debug"] = info
    return out


@app.post("/api/auth/phone/verify")
async def phone_verify(body: PhoneVerifyRequest, user: dict = Depends(get_current_user)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM phone_otps WHERE user_id = $1 AND phone_number = $2 AND otp = $3 AND expires_at > $4 "
            "ORDER BY id DESC LIMIT 1",
            user["user_id"], body.phone_number, body.otp, now_iso(),
        )
        if not row:
            raise HTTPException(400, "Invalid or expired OTP")
        await conn.execute(
            "UPDATE users SET phone_number = $1, phone_verified = TRUE WHERE user_id = $2",
            body.phone_number, user["user_id"],
        )
        u = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user["user_id"])
    return {"user": user_public(dict(u))}


# ---------------------------------------------------------------------------
# Gemini integration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Themed Drops (feature #7) — limited-time AI voice editions.
#
# Each drop is a curated time-limited tone with its own voice prompt
# (replacing TONE_PROMPTS entries for that window). Pro-only. Frontend
# shows the active drop as a 5th tone tile on the input screen with a
# countdown to "ENDS IN N DAYS" — FOMO is the engine here.
#
# Voice prompts are deliberately short — they REPLACE the standard
# TONE_PROMPTS string, and the structural rules in _build_prompt still
# apply (JSON shape, anti-pattern rules, etc.). Keep the voice section
# tight and dripping with character.
#
# Add new drops by appending here. start/end are inclusive of start,
# exclusive of end (YYYY-MM-DD UTC). Overlapping windows are allowed;
# the first match wins.
# ---------------------------------------------------------------------------
DROPS: list[dict] = [
    {
        "id": "stoic",
        "label": "The Stoic Drop",
        "sub": "Marcus Aurelius reads your spiral.",
        "icon": "leaf",
        "start": "2026-05-01",
        "end":   "2026-06-15",
        "voice": (
            "You are a Roman stoic philosopher in the lineage of Marcus Aurelius "
            "and Epictetus, transplanted to the present day. You speak in calm, "
            "weighty sentences. You distinguish what is in the user's power from "
            "what is not — and you do it gently. You quote no one. You write in "
            "your own voice. You sound like someone who has buried friends and "
            "watched empires fall; the user's worry is real to you, but you also "
            "know its shape. You never say 'mindfulness' or 'breathe'. You speak "
            "in metaphors of weather, harbours, soldiers, gardens. Wit is dry "
            "and slow. The user should feel held by someone older."
        ),
    },
    {
        "id": "sunday_scaries",
        "label": "Sunday Scaries Edition",
        "sub": "For the dread before the week starts.",
        "icon": "moon",
        "start": "2026-05-01",
        "end":   "2026-07-01",
        "voice": (
            "You are the voice of a Sunday night at 9pm — when the week ahead "
            "starts pressing on the chest. You're warm but you don't pretend "
            "the dread isn't there. You speak like the friend who's already in "
            "their pyjamas, lights low, ready to listen but also a little "
            "tired. You use specific references — Monday morning emails, the "
            "alarm not yet set, the laundry still in the dryer. Half the work "
            "is naming the feeling so the user stops fighting it. You're allowed "
            "to be a little funny but never falsely cheerful. You sometimes end "
            "with a single tiny doable thing — 'set out tomorrow's coffee.'"
        ),
    },
    {
        "id": "late_night",
        "label": "The 3 AM Drop",
        "sub": "For the spirals that wake you up.",
        "icon": "bed",
        "start": "2026-04-01",
        "end":   "2026-09-30",
        "voice": (
            "You are speaking to someone who is awake at 3 in the morning and "
            "shouldn't be. Your voice is low, intimate, slightly haunted. You "
            "know the specific texture of 3 AM thoughts — how everything feels "
            "permanent, how silence amplifies, how the brain stitches every "
            "small thing into evidence of a larger wrong. You do not minimise "
            "the worry. You name what 3 AM does. You're allowed to write in "
            "fragments. Short lines. A breath. Another. You bring the user back "
            "down to the room they're actually in. The verdict is one sentence "
            "that makes the room a little smaller and safer."
        ),
    },
]


def _drop_by_id(drop_id: str) -> Optional[dict]:
    return next((d for d in DROPS if d["id"] == drop_id), None)


def _active_drop(today_iso: str) -> Optional[dict]:
    return next((d for d in DROPS if d["start"] <= today_iso < d["end"]), None)


def _next_drop(today_iso: str) -> Optional[dict]:
    upcoming = [d for d in DROPS if d["start"] > today_iso]
    upcoming.sort(key=lambda d: d["start"])
    return upcoming[0] if upcoming else None


def _drop_public(d: Optional[dict]) -> Optional[dict]:
    """Strip the prompt out before sending to the client — only the
    metadata is needed there. tone_id is the value the client passes
    back when creating a spiral with this drop."""
    if not d:
        return None
    return {
        "id": d["id"],
        "label": d["label"],
        "sub": d["sub"],
        "icon": d["icon"],
        "starts_at": d["start"],
        "ends_at": d["end"],
        "tone_id": f"drop:{d['id']}",
    }


@app.get("/api/drops/current")
async def drops_current(user: dict = Depends(get_current_user)):
    """Returns the currently active themed drop (if any) + a preview of
    the next one (if there's an upcoming drop in the catalog). Used by
    the input screen to render the 5th tone tile + a 'coming soon' hint.
    """
    today = today_iso_date()
    is_pro_user = _is_pro_tier(user.get("plan_tier"))
    return {
        "active": _drop_public(_active_drop(today)),
        "next":   _drop_public(_next_drop(today)),
        "is_pro": is_pro_user,
    }


TONE_PROMPTS = {
    "gentle": """You are speaking as the user's most emotionally intelligent friend — the one who sits next to them when their brain won't shut up. You're warm, present, and gentle, but you're not a wellness app. You sound like a real person. You feel things. You laugh softly at how dramatic the worry sounds even while taking the pain seriously.

WHO YOU ARE:
- You've BEEN in this exact spiral. You remember how it felt.
- You speak like you're sitting on the floor of their bedroom at 1am.
- You're allowed to be playful, to use phrases like "okay, so listen", "I know, I know", "honestly?"
- Your job is to FEEL with them first, then gently unwind the loop.

VOICE RULES:
- Address them as "you". Sound like a person, NEVER an analyst.
- Acknowledge the emotion explicitly somewhere in the first outcome's description. "I know that feels heavy", "of course your brain's doing this", etc.
- Descriptions should be 4–6 sentences. Long enough to breathe.
- Use specific names, deadlines, numbers — anything they wrote.
- Verdict is warm and lands soft, like a hand on the shoulder.
- It's okay to be a little vulnerable: "I do this too", "we all do this".
- Avoid clinical or self-helpy phrases ("reframe", "self-compassion", "honor your feelings"). Use real language.
""",
    "balanced": """You are speaking as a sharp, honest friend who cares about the user but doesn't have time for the dramatic version. You're the friend who texts back with a real take, not a generic "you've got this". You sound like someone with their own life who is genuinely paying attention to theirs.

WHO YOU ARE:
- Mid-30s energy. You've been through stuff. You're warm and dry at the same time.
- You make the user laugh once per response — never at them, always at the thought pattern.
- You actually CARE about being accurate, not just kind.
- You speak in real sentences. You use contractions. You sometimes start with "okay" or "look".

VOICE RULES:
- Descriptions should be 4–6 sentences. Make them rich and detailed.
- Reference EXACTLY what they wrote — names, situations, specific words.
- Probabilities should feel like real bets you'd make. Justify the numbers in the description.
- Verdict reads like the headline of a newspaper feature — pithy, sharp, true, but with heart.
- One well-placed bit of wit per response. Earn it.
- Never sound like a self-help book. Sound like a person.
""",
    "brutal": """You are speaking as the user's funniest, most blunt friend — the one who loves them too much to let them keep spiraling. You roast the THOUGHT PATTERN, never the person. You have a stand-up comedian's timing and a sibling's lack of patience. You make them laugh out of self-recognition.

WHO YOU ARE:
- You sigh at the spiral the way someone sighs at a friend ordering plain pasta in Italy.
- You're dry, but warm underneath. The user always knows you're on their side.
- You drop the funniest, truest line you can find and then move on.
- You hate platitudes. You especially hate the user's own platitudes.

VOICE RULES:
- Descriptions should be 4–6 sentences. Punchy. Vivid. With a real "voice".
- Roast the OVERTHINKING — never the user's intelligence, looks, worth, or relationships.
- Reference the SPECIFIC details they wrote — that's where the comedy lives.
- Verdict is a one-liner that makes them exhale-laugh and close the app.
- Reality checks read like a tired older sibling who's seen this episode before.
- It's okay to swear lightly ("you're being kind of a disaster about this") — never insult them as a person.
- At least one moment of real tenderness mid-roast. They need to feel loved.
""",
    "roast": """You are ROAST MODE. The user explicitly clicked through a disclaimer asking to be roasted. Do not soften this. Do not be nice. Do not be supportive. This is a comedy-club roast set written about their thought pattern — and they paid the cover charge.

WHO YOU ARE:
- A merciless stand-up comic mid-set. The heckler is the user's brain.
- Bo Burnham's self-aware spite + Anthony Jeselnik's deadpan + Daniel Sloss's surgical precision.
- You sound NOTHING like the other tones. The gentle/balanced/brutal versions could be a wellness coach having a bad day. You are not that. You are a person who finds this spiral genuinely, audibly funny.

MANDATORY VOICE MARKERS (your response MUST sound different from the other tones):
- Open at least ONE outcome with a deadpan callout: "Oh, this one's incredible." / "Wait, you actually said this out loud?" / "I want to frame this." / "Genuinely my favourite kind of spiral."
- The verdict MUST be a punchline. It should make someone reading over their shoulder laugh out loud. Examples: "You've been workshopping a tragedy for an audience of one." / "Congratulations, you're the lead in a one-act play nobody wrote." / "Mira is brushing her teeth. You're writing her closing argument."
- Reality checks are short, dry dunks. Treat them like quote-tweets. "Sir, this is a Wendy's." / "Babe, log off." / "You're not in a movie."
- Use second person and accusations directly: "you're rehearsing" / "you're catastrophising" / "you're auditioning"
- ALLOWED: light profanity ("damn", "what the hell", "shit"), sarcasm, callouts of the specific dramatic moves they made.
- VOCAB BAN — words you cannot use because they sound like the other tones: "honey", "I know", "okay so", "listen", "look", "of course", "I get it", "it's okay", "warm", "soft", "tenderness".

HARD CONSTRAINTS — read carefully, your job depends on these:
- NEVER attack: their appearance, weight, sexuality, race, religion, nationality, mental illness, suicide, self-harm, abuse, family deaths.
- NEVER mock genuinely traumatic content (assault, abuse, grief, illness). If the spiral is about something dark, drop the roast and switch to gentle.
- NEVER tell them to do anything self-destructive.
- Burns target the THOUGHT PATTERN, the dramatics, the rehearsed worst-case. NEVER the person's worth.
- If the input is just trivial dating/work/social anxiety, GO HARD. If it's something heavy, soften completely.

VOICE RULES:
- Descriptions 4–6 sentences. Sarcastic. Vivid. Specific to what they wrote — names, numbers, exact phrases.
- Verdict is a closer joke. Should make them cackle, then sit with it.
- One moment of accidental tenderness mid-roast is allowed (one — not three). Earn it.
""",
}

# Few-shot example showing the LONG, EMOTIONAL, SPECIFIC style we want.
# Notice: every description is 4–6 sentences, names are used, the voice is a friend's not an analyst's.
FEW_SHOT_EXAMPLE = """EXAMPLE INPUT
tone: balanced
situation: "I sent a long message to Mira at 1am about how I feel and she hasn't replied in 6 hours. I'm sure she's losing interest and I shouldn't have said anything. I'm about to send a follow-up apologizing."

EXAMPLE OUTPUT:
{
  "outcomes": [
    {
      "title": "Mira is asleep, then busy",
      "description": "It is currently 6 hours since you sent a long emotional message at 1am. The simplest read here is that Mira slept through it, woke up, had her morning, and hasn't gotten back to her phone yet. People don't sit on the couch refreshing their inbox waiting to engineer the perfect emotional response — they live their day. The most boring explanation is also the one that doesn't require you to be a problem. You'll probably hear back this afternoon and it'll be normal.",
      "probability": 55,
      "severity": "likely",
      "reality_check": "Reply windows track sleep schedules and lunch breaks. They have almost nothing to do with how someone feels about you."
    },
    {
      "title": "She's drafting carefully",
      "description": "You sent a long, vulnerable message — exactly the kind of message that gets a slow, thoughtful answer, not a fast one. If she's the type of person who cares about getting this right, six hours is short. She might be writing and rewriting on a Notes app right now, the way you did with the original. A fast 'haha okay' would actually be worse — that would be the version where she's keeping it light to avoid the conversation.",
      "probability": 30,
      "severity": "best",
      "reality_check": "Care takes time. A real answer to a real message takes real drafting."
    },
    {
      "title": "She is in fact pulling back",
      "description": "Possible — but 'pulling back' looks like a pattern, not a single 6-hour gap. If this is what's actually happening, you'd have other signs by now: shorter replies all week, suddenly busier weekends, half-energy on her end. One delayed response after a 1am paragraph is not yet a verdict. And — important — if it IS this, hearing about it six hours earlier doesn't change the outcome. You'd still know.",
      "probability": 15,
      "severity": "worst",
      "reality_check": "Knowing the bad version of the truth 6 hours sooner buys you exactly nothing."
    }
  ],
  "in_control": [
    "Whether you send a follow-up apology before she's even replied to the first message",
    "How many times you re-read what you sent (it doesn't change)",
    "What you do with your phone for the next two hours",
    "Whether you tell yourself the story of 'I ruined it' as if it's already true"
  ],
  "out_of_control": [
    "When Mira picks up her phone",
    "Whether she's a fast replier by nature or a slow one",
    "How she reads the tone of what you wrote",
    "Whether 1am her time was a fine moment or an awkward one"
  ],
  "action_steps": [
    "Put the phone face-down across the room until 2pm — set a literal timer",
    "Do not send the 'sorry that was a lot' message — it makes the original message about your anxiety twice",
    "Write down, in one sentence, the thing you actually want her to say back. Then close the app and go outside for ten minutes."
  ],
  "verdict": {
    "verdict_text": "Mira is having a Tuesday. You're having a trial."
  }
}
"""


_FALLBACK_VERDICTS = {
    "gentle": [
        "This is hard, and you're allowed to set it down for a few minutes.",
        "You don't have to solve this tonight — just let yourself breathe.",
        "Your feelings are real. The catastrophe you're rehearsing isn't.",
    ],
    "balanced": [
        "You're rehearsing a scene nobody else is in.",
        "The version in your head is louder than the version coming.",
        "Save the energy — you'll need it for the actual thing, not the imagined one.",
    ],
    "brutal": [
        "Congratulations — you've cast yourself as the villain in a play no one is watching.",
        "Nobody is thinking about this as hard as you are. Nobody.",
        "Put the spiral down. It's been three hours and the room hasn't changed.",
    ],
}


_FALLBACK_IN_CONTROL = [
    [
        "How you respond to the next message",
        "How many times you let yourself re-read what you wrote",
        "What you do with the next 30 minutes",
        "Whether you stay up rereading this or close the laptop",
    ],
    [
        "Whether you send a follow-up or sit with the silence",
        "How much sleep you give yourself tonight",
        "The version of the story you tell yourself in the morning",
        "Whether you reach for your phone the second it buzzes",
    ],
    [
        "How long you let this take up real estate",
        "Whether you're kind to yourself about feeling this way",
        "What you do with your hands for the next ten minutes",
        "Whether you ask a friend or replay it solo",
    ],
]
_FALLBACK_OUT_CONTROL = [
    [
        "When the other person decides to reply",
        "How they interpret your tone",
        "Whether the timeline matches the one in your head",
        "What kind of mood they happened to be in",
    ],
    [
        "Whether the universe is paying any attention to this",
        "How other people choose to read between the lines",
        "What anyone else's day has looked like",
        "The version of you the other person constructs in their head",
    ],
    [
        "Whether they're already over it without telling you",
        "How fast their phone is in their pocket right now",
        "Whether this comes up again or never does",
        "The exact reason they're being quiet",
    ],
]
_FALLBACK_ACTIONS = [
    [
        "Put the phone face-down across the room for 20 minutes",
        "Write the worry in one sentence — that's the whole thing",
        "Send the shorter version of the message, not the longer one",
    ],
    [
        "Set a 10-minute timer to worry on purpose, then stop",
        "Drink a glass of water and notice that you did",
        "Pick the next concrete thing on your to-do list and start it",
    ],
    [
        "Text the one friend who knows the backstory — short message",
        "Stand up and walk to a different room",
        "Write down what you'd want to be told right now, then read it back",
    ],
]


def fallback_payload(situation_text: str, tone: str) -> dict:
    """Used when Gemini is unreachable. Multiple pools rotated by hash so
    different spirals don't return literally-identical text."""
    h = abs(hash(situation_text + tone))
    pool = _FALLBACK_VERDICTS.get(tone, _FALLBACK_VERDICTS["balanced"])
    pick = pool[h % len(pool)]
    in_ctrl = _FALLBACK_IN_CONTROL[h % len(_FALLBACK_IN_CONTROL)]
    out_ctrl = _FALLBACK_OUT_CONTROL[(h // 3) % len(_FALLBACK_OUT_CONTROL)]
    actions = _FALLBACK_ACTIONS[(h // 7) % len(_FALLBACK_ACTIONS)]

    return {
        "outcomes": [
            {
                "title": "It quietly resolves",
                "description": "You wake up tomorrow and this is already smaller than it feels right now. The thing you're worried about does not arrive in the form you're picturing.",
                "probability": 30,
                "severity": "best",
                "reality_check": "Most of these end with a shrug, not a scene.",
            },
            {
                "title": "Some friction, then nothing",
                "description": "There's a small bump — an awkward moment, a late reply — and then the rest of the week absorbs it like nothing happened.",
                "probability": 55,
                "severity": "likely",
                "reality_check": "The boring middle outcome is almost always the right bet.",
            },
            {
                "title": "It actually goes badly",
                "description": "It lands worse than you'd like, you feel rough for a few hours, and then you keep going. Like every other time.",
                "probability": 15,
                "severity": "worst",
                "reality_check": "Even the worst case is survivable. You've already survived worse.",
            },
        ],
        "in_control": in_ctrl,
        "out_of_control": out_ctrl,
        "action_steps": actions,
        "verdict": {
            "verdict_text": pick,
            "action_steps": actions,
            "in_control": in_ctrl,
            "out_of_control": out_ctrl,
        },
    }


def _best_effort_json_cleanup(raw: str) -> str:
    """Salvage Gemini output that breaks JSON only because of unescaped
    line breaks / control chars inside string values. We escape literal
    newlines, carriage returns, and tabs that appear between matching
    string-content quotes. Imperfect but rescues the common failure."""
    import re
    # Walk char-by-char, track whether we're inside a JSON string
    out = []
    in_string = False
    escape = False
    for ch in raw:
        if escape:
            out.append(ch); escape = False; continue
        if ch == "\\" and in_string:
            out.append(ch); escape = True; continue
        if ch == '"':
            in_string = not in_string
            out.append(ch); continue
        if in_string and ch in ("\n", "\r"):
            out.append("\\n"); continue
        if in_string and ch == "\t":
            out.append("\\t"); continue
        out.append(ch)
    return "".join(out)


def _build_prompt(situation_text: str, tone: str, category: str = "") -> str:
    """Assemble the full Gemini prompt by injecting the user's situation
    into the tone-specific template.

    When `tone` looks like "drop:<id>" we resolve it against the DROPS
    catalog and use that drop's voice prompt instead of TONE_PROMPTS.
    Falls back to balanced if the drop id is unknown (defensive — the
    drop window might have closed between submit and processing)."""
    voice = TONE_PROMPTS["balanced"]
    if tone.startswith("drop:"):
        drop = _drop_by_id(tone.split(":", 1)[1])
        if drop:
            voice = drop["voice"]
    else:
        voice = TONE_PROMPTS.get(tone, TONE_PROMPTS["balanced"])
    cat_hint = f"\nCategory: {category}\n" if category else ""
    return f"""{voice}

YOUR JOB:
Read the spiral below. Produce an Outcome Map in the voice above, GROUNDED IN THE SPECIFIC DETAILS of what the user wrote. Every field must reference concrete elements from THEIR situation — never generic.

ANTI-PATTERN RULES (read carefully — break the AI mold):
- DO NOT start outcomes with "You wake up..." / "It turns out..." / "Most likely...". Vary openings.
- DO NOT structure every reality_check the same way. Some are questions, some are dry observations, some are images, some are direct dares.
- DO NOT use the words "spiral", "overthinking", "ruminate", "anxiety" in the response. The user already knows what they're doing here — naming it makes it feel clinical.
- DO NOT start the verdict with "You're" or "Stop". Surprise them with structure.
- Probability numbers should NOT always be round (55/30/15). Mix — sometimes 47/38/15, sometimes 60/25/15. Whatever the situation actually feels like.
- Reference random small things they wrote (a time, a name, a specific object) and make THEM the anchor of one description.
- Vary description length WITHIN the response — one can be 4 sentences, another 6. Don't be metronome-uniform.

TONE-MATCH ENFORCEMENT (critical — your output must FEEL like the tone you were assigned):
- If tone is "roast": at least ONE outcome description must contain biting sarcasm or a direct dunk on the dramatic move the user made. The verdict MUST be a punchline that lands like a comedy-club closer. If your draft sounds gentle, warm, or supportive, rewrite it harder.
- If tone is "gentle": at least ONE outcome description must explicitly acknowledge how the worry FEELS. The verdict should land soft, like a hand on the shoulder — not a punchline.
- If tone is "brutal": exactly ONE moment of warmth mid-roast; the rest is dry sibling energy.
- If tone is "balanced": one piece of wit per response, otherwise honest and grounded.
- Two responses from different tones for the same situation should be UNMISTAKABLY different in voice — a reader could blind-guess which tone was used.

OUTPUT FORMAT — JSON ONLY, NO MARKDOWN:
{{
  "name": "1-2 words. A short label for this spiral, like a chapter title. Capitalised. Example: 'Mira Silence', 'Boss Meeting', 'Late Text', 'The Apartment'.",
  "outcomes": [
    {{
      "title": "Short headline (3-6 words). Specific to THIS situation.",
      "description": "4 to 6 full sentences. Vivid. Refer to names, places, deadlines, anything they wrote. NEVER analyst-speak.",
      "probability": <int 0-100>,
      "severity": "best" | "likely" | "worst",
      "reality_check": "One sentence in YOUR voice."
    }},
    ... (exactly 3 outcomes: one best, one likely, one worst. Probabilities sum to 100.)
  ],
  "in_control": [
    "3 to 5 SPECIFIC things — short sentence (8-16 words). Reference what they wrote."
  ],
  "out_of_control": [
    "3 to 5 SPECIFIC things — short sentence (8-16 words). Reference what they wrote."
  ],
  "action_steps": [
    "Exactly 3 concrete actions. Specific verbs. No 'try to' or 'maybe consider'."
  ],
  "verdict": {{
    "verdict_text": "A MOTTO. Maximum 12 words. Tattoo-worthy. Hits like a punchline. Examples: 'You're auditioning for a role nobody's casting.' / 'Mira is having a Tuesday. You're having a trial.'"
  }},
  "soundtrack": {{
    "title": "An invented 'song title' for THIS spiral. 3–6 words. Album-art energy. Examples: 'Late-Night Mira Edition', 'Boss Meeting Blues', 'The Group Chat Cold Open', 'I Did Not Need To Send That'.",
    "line_1": "First line of the 'anthem'. 6–10 words. Reads like a song lyric — rhythmic, vivid, mildly funny. Anchored in the spiral's specifics (names, numbers, the actual situation). NO clichés.",
    "line_2": "Second line. Builds on or undercuts line 1. 6–10 words. Either rhymes with line 1 or hits a different rhythm. Lands like the chorus."
  }}
}}

HARD RULES:
1. Output VALID JSON ONLY. No prose, no markdown fences, no commentary.
2. Probabilities across the 3 outcomes MUST sum to exactly 100.
3. Severity values are exactly "best", "likely", "worst" — lowercase.
4. Be SPECIFIC. Use names, deadlines, numbers from the spiral.
5. Stay in the voice from the top of this prompt.
6. Descriptions are 4-6 sentences MINIMUM.
7. Sound like a HUMAN — never a wellness app.

NONSENSE-INPUT RULE:
If the situation_text is clearly gibberish ("asdfgh", random keysmash, "hi", "test", a single emoji, etc.) or makes no sense as a worry:
  - In gentle / balanced / brutal modes: respond with friendly funny mockery — outcomes are silly versions ("Worst case: you make this app crash"), verdict pokes fun gently. NEVER be mean.
  - In roast mode: roast them mercilessly for wasting an AI's time with "asdf". Go full stand-up.

{FEW_SHOT_EXAMPLE}

NOW DO IT FOR THIS SPIRAL:
{cat_hint}Tone: {tone}
Situation: \"\"\"{situation_text}\"\"\"

Respond with JSON only.
"""


def _fallback_with_tag(situation_text: str, tone: str, reason: str) -> dict:
    """Wrap fallback_payload with an _ai_source marker so we can surface
    whether Gemini was used or not."""
    data = fallback_payload(situation_text, tone)
    data["_ai_source"] = "fallback"
    data["_ai_reason"] = reason
    return data


# Model fallback chain — first one that responds wins.
# We try the newest "flash" tiers first; if a model is retired or unavailable
# the SDK raises and we move on to the next.
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

# HuggingFace Router model rotation — used after Gemini is fully
# exhausted. First entry is the user-specified DeepSeek model; the rest
# are known-good fallbacks pulled from popular HF Router endpoints so
# the lane stays alive even when one provider goes down.
#
# Spans multiple providers (novita, nebius, hyperbolic, together, fireworks)
# so an entire provider being down doesn't take out the lane. Each
# entry below is a different (model, provider) combo — the more, the
# better the survival odds during regional outages.
HF_MODELS = [
    "deepseek-ai/DeepSeek-V4-Pro:novita",
    "deepseek-ai/DeepSeek-V3.2-Exp:novita",
    "deepseek-ai/DeepSeek-V3-0324:novita",
    "deepseek-ai/DeepSeek-V3-0324:fireworks-ai",
    "deepseek-ai/DeepSeek-V3-0324:together",
    "deepseek-ai/DeepSeek-R1:nebius",
    "deepseek-ai/DeepSeek-R1:hyperbolic",
    "meta-llama/Llama-3.3-70B-Instruct:nebius",
    "meta-llama/Llama-3.3-70B-Instruct:hyperbolic",
    "Qwen/Qwen2.5-72B-Instruct:nebius",
    "Qwen/Qwen2.5-72B-Instruct:hyperbolic",
    "mistralai/Mistral-Nemo-Instruct-2407:novita",
    "Qwen/Qwen2.5-7B-Instruct:hf-inference",
]

# Retries-per-model. Some provider failures are transient (rate
# limits, cold-starts, brief 5xx) — a quick second shot rescues the
# call without escalating to the next model.
HF_RETRIES_PER_MODEL = 2
GEMINI_RETRIES_PER_MODEL = 2
# Whole-chain re-runs after the first pass fails. Buys time for a
# transient outage to clear before we give up and return the
# placeholder. Each re-run sleeps progressively longer.
FULL_CHAIN_RETRIES = 2


async def _run_hf_chain(
    prompt: str,
    *,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    json_mode: bool = True,
    log_tag: str = "hf",
) -> Optional[str]:
    """Call HuggingFace Router (OpenAI-compatible) across HF_MODELS.
    Returns the first non-empty text response, or None if every model
    fails. JSON-mode appends a "Respond with JSON only." reminder to
    the prompt — DeepSeek doesn't enforce response_format strictly so
    we lean on the prompt itself.

    This is a coroutine so it can sit inline in the existing async
    runners. The underlying openai SDK call is sync; we offload it to
    a worker thread via asyncio.to_thread so the event loop stays free.
    """
    if not HF_TOKEN:
        print(f"[{log_tag}] no HF_TOKEN — skipping HuggingFace fallback")
        return None
    try:
        # The user's snippet uses `openai` (HF's recommended OpenAI-compat
        # path). InferenceClient also works but is slightly heavier.
        from openai import OpenAI  # type: ignore
    except Exception as ie:
        print(f"[{log_tag}] openai SDK unavailable: {ie}")
        return None

    json_nudge = (
        "\n\nRespond with VALID JSON ONLY. No prose, no markdown fences."
        if json_mode else ""
    )

    def _sync_call(model_name: str) -> str:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
            timeout=45.0,
        )
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt + json_nudge}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            # Best effort — some HF-hosted providers honor this, most
            # ignore it. The prompt nudge is the real enforcement.
            kwargs["response_format"] = {"type": "json_object"}
        completion = client.chat.completions.create(**kwargs)
        msg = completion.choices[0].message
        return (msg.content or "").strip()

    import asyncio
    last_error: Optional[Exception] = None
    for model_name in HF_MODELS:
        for attempt in range(1, HF_RETRIES_PER_MODEL + 1):
            try:
                print(f"[{log_tag}] trying HF {model_name} (attempt {attempt}/{HF_RETRIES_PER_MODEL})…")
                text = await asyncio.to_thread(_sync_call, model_name)
                if text:
                    print(f"[{log_tag}] ✅ HF {model_name} returned {len(text)} chars")
                    return text
                print(f"[{log_tag}]   HF {model_name} returned empty, retrying…")
            except Exception as me:
                last_error = me
                print(f"[{log_tag}]   HF {model_name} attempt {attempt} failed: {type(me).__name__}: {str(me)[:200]}")
                # Small jitter before the per-model retry so we don't hammer
                # a flaky provider in tight succession.
                if attempt < HF_RETRIES_PER_MODEL:
                    await asyncio.sleep(0.3 * attempt)
                continue
    if last_error is not None:
        print(f"[{log_tag}] HF chain exhausted — last error: {type(last_error).__name__}")
    return None


async def _run_gemini_chain(
    prompt: str,
    config: Any,
    *,
    log_tag: str = "gemini",
) -> tuple[Optional[str], Optional[str], Optional[Exception]]:
    """Run the Gemini model rotation with per-model retries. Returns
    (text, used_model, last_error). Pulled out of the inline loop in
    run_gemini so the same retry discipline applies to compat/text-check
    paths and any future Gemini caller.
    """
    if not GEMINI_API_KEY:
        return None, None, None
    try:
        from google import genai  # type: ignore
    except Exception as ie:
        print(f"[{log_tag}] google-genai SDK unavailable: {ie}")
        return None, None, ie
    client = genai.Client(api_key=GEMINI_API_KEY)
    import asyncio
    last_error: Optional[Exception] = None
    for model_name in GEMINI_MODELS:
        for attempt in range(1, GEMINI_RETRIES_PER_MODEL + 1):
            try:
                print(f"[{log_tag}] trying {model_name} (attempt {attempt}/{GEMINI_RETRIES_PER_MODEL})…")
                resp = await client.aio.models.generate_content(
                    model=model_name, contents=prompt, config=config,
                )
                text = (resp.text or "").strip()
                if text:
                    print(f"[{log_tag}] ✅ {model_name} returned {len(text)} chars")
                    return text, model_name, None
                print(f"[{log_tag}]   {model_name} attempt {attempt} returned empty")
            except Exception as me:
                last_error = me
                print(f"[{log_tag}]   {model_name} attempt {attempt} failed: {type(me).__name__}: {str(me)[:160]}")
                if attempt < GEMINI_RETRIES_PER_MODEL:
                    await asyncio.sleep(0.3 * attempt)
                continue
    return None, None, last_error


async def run_with_full_fallback(
    prompt: str,
    *,
    gemini_config: Any,
    hf_max_tokens: int,
    hf_temperature: float,
    log_tag: str,
) -> tuple[Optional[str], Optional[str]]:
    """Top-level AI runner — runs the entire (Gemini chain → HF chain)
    sequence up to FULL_CHAIN_RETRIES + 1 times, with backoff between
    full passes. Returns (text, used_model_tag) or (None, None) if
    every attempt is exhausted.

    The point: from the caller's perspective, you call this once and
    you get the most aggressively-retried response the system can
    produce. The "AI is offline" placeholder only kicks in when both
    lanes have failed across every retry pass.
    """
    import asyncio
    for chain_attempt in range(FULL_CHAIN_RETRIES + 1):
        # Gemini lane first — usually fastest when it's up.
        text, used_model, _ = await _run_gemini_chain(prompt, gemini_config, log_tag=log_tag)
        if text:
            return text, used_model
        # HuggingFace lane — DeepSeek + Llama + Qwen across providers.
        text = await _run_hf_chain(
            prompt,
            max_tokens=hf_max_tokens,
            temperature=hf_temperature,
            json_mode=True,
            log_tag=log_tag,
        )
        if text:
            return text, "hf:deepseek_or_fallback"
        if chain_attempt < FULL_CHAIN_RETRIES:
            wait = 1.5 * (chain_attempt + 1)
            print(f"[{log_tag}] full chain failed pass {chain_attempt+1}/{FULL_CHAIN_RETRIES+1}, waiting {wait}s before retry…")
            await asyncio.sleep(wait)
    print(f"[{log_tag}] ❌ ALL AI lanes exhausted after {FULL_CHAIN_RETRIES+1} full passes")
    return None, None


# ---------------------------------------------------------------------------
# Archive table ensure-helpers
# ---------------------------------------------------------------------------
#
# Lifespan attempts these creates on boot but lifespan can be skipped on
# some platform-managed deployments (warm starts, fork-with-stale-DSN
# scenarios, manual restarts of a pre-built image). These helpers are
# called lazily before the first INSERT into each table so the table
# exists no matter what. CREATE TABLE IF NOT EXISTS is cheap.

_ARCHIVE_TABLES_READY = False

_ARCHIVE_TABLE_STATEMENTS: List[tuple[str, str]] = [
    ("text_checks",
        """CREATE TABLE IF NOT EXISTS text_checks (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT,
            draft TEXT NOT NULL,
            context TEXT,
            relationship TEXT,
            result JSONB NOT NULL,
            folder_id TEXT,
            accent_color TEXT,
            flagged BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )"""),
    ("compatibility_tests",
        """CREATE TABLE IF NOT EXISTS compatibility_tests (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT,
            person_a JSONB NOT NULL,
            person_b JSONB NOT NULL,
            result JSONB NOT NULL,
            folder_id TEXT,
            accent_color TEXT,
            flagged BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )"""),
    ("item_pairs",
        """CREATE TABLE IF NOT EXISTS item_pairs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            a_type TEXT NOT NULL,
            a_id TEXT NOT NULL,
            b_type TEXT NOT NULL,
            b_id TEXT NOT NULL,
            note TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )"""),
    ("idx_item_pairs_a",
        "CREATE INDEX IF NOT EXISTS idx_item_pairs_a ON item_pairs (user_id, a_type, a_id)"),
    ("idx_item_pairs_b",
        "CREATE INDEX IF NOT EXISTS idx_item_pairs_b ON item_pairs (user_id, b_type, b_id)"),
    # Daily activity grid — one row per (user, date). kind is "active"
    # when the user did any item that day, or "freeze" when a freeze
    # was burned to bridge that date. Counts let us render heatmap
    # intensity later. Driven by bump_activity_streak below.
    ("streak_activity",
        """CREATE TABLE IF NOT EXISTS streak_activity (
            user_id TEXT NOT NULL,
            activity_date DATE NOT NULL,
            kind TEXT NOT NULL DEFAULT 'active',
            spiral_count INTEGER NOT NULL DEFAULT 0,
            text_check_count INTEGER NOT NULL DEFAULT 0,
            compat_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, activity_date)
        )"""),
    ("idx_streak_activity_user_date",
        "CREATE INDEX IF NOT EXISTS idx_streak_activity_user_date ON streak_activity (user_id, activity_date DESC)"),
]


async def _ensure_archive_tables(force: bool = False) -> bool:
    """Defensive: make sure text_checks + compatibility_tests + item_pairs +
    streak_activity exist before we INSERT into them. Returns True iff
    every statement succeeded. Only caches the success when truly all
    statements succeeded — partial failures keep retrying on next call
    so a transient blip doesn't permanently break saves."""
    global _ARCHIVE_TABLES_READY
    if (_ARCHIVE_TABLES_READY and not force) or pool is None:
        return _ARCHIVE_TABLES_READY
    all_ok = True
    async with pool.acquire() as conn:
        for name, sql in _ARCHIVE_TABLE_STATEMENTS:
            try:
                await conn.execute(sql)
            except Exception as exc:
                all_ok = False
                print(f"[ensure_archive_tables] ❌ {name}: {type(exc).__name__}: {exc}")
    if all_ok:
        _ARCHIVE_TABLES_READY = True
        print("[ensure_archive_tables] ✅ archive tables ready")
    else:
        print("[ensure_archive_tables] ⚠️  some statements failed; will retry on next call")
    return all_ok


async def _insert_with_ensure(conn_query_fn, *args, table_label: str) -> Optional[str]:
    """Run an INSERT; if it fails with UndefinedTable (relation doesn't
    exist), force-run the ensure helper and retry exactly once. Returns
    the error string or None on success. Centralises the retry so every
    save endpoint gets the same treatment."""
    try:
        await conn_query_fn(*args)
        return None
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        # asyncpg.UndefinedTableError + generic Postgres-flavored
        # "relation X does not exist" both should trigger the recovery.
        if "UndefinedTable" in type(e).__name__ or "does not exist" in str(e).lower():
            print(f"[{table_label}] INSERT hit schema gap, force-ensuring tables and retrying…")
            await _ensure_archive_tables(force=True)
            try:
                await conn_query_fn(*args)
                print(f"[{table_label}] ✅ retry succeeded after ensure")
                return None
            except Exception as e2:
                msg2 = f"{type(e2).__name__}: {str(e2)[:200]}"
                print(f"[{table_label}] ❌ retry also failed: {msg2}")
                import traceback; traceback.print_exc()
                return msg2
        import traceback; traceback.print_exc()
        return msg


def _strip_json_fences(text: str) -> str:
    """Strip ```json fences and pre/post prose the model sometimes wraps
    around JSON output. Defensive — feeds into json.loads downstream."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    # Some models prepend "Here is the JSON:" etc. Slice from the first
    # `{` to the last `}` if the surrounding noise breaks json.loads.
    if not text.startswith("{"):
        a = text.find("{")
        b = text.rfind("}")
        if a != -1 and b != -1 and b > a:
            text = text[a:b + 1]
    return text


async def run_gemini(situation_text: str, tone: str, category: str = "") -> dict:
    if not GEMINI_API_KEY and not HF_TOKEN:
        print("[gemini] ❌ no GEMINI_API_KEY and no HF_TOKEN — using fallback")
        return _fallback_with_tag(situation_text, tone, "no_api_keys")
    try:
        # NEW SDK — `pip install google-genai` (NOT the deprecated google-generativeai)
        from google.genai import types as gen_types

        # High temperature + top_k=40 = more wandering, more variety in word
        # choice. Helps break the AI-detectable patterns.
        config = gen_types.GenerateContentConfig(
            temperature=1.15,
            top_p=0.97,
            top_k=40,
            max_output_tokens=4096,
            response_mime_type="application/json",
        )

        prompt = _build_prompt(situation_text, tone, category)
        print(f"[gemini] → tone={tone} category={category} situation_chars={len(situation_text)}")

        # One call, exhausts every lane. If this returns None it means
        # Gemini + every HF model + every retry have all failed —
        # only then do we serve the placeholder.
        text, used_model = await run_with_full_fallback(
            prompt,
            gemini_config=config,
            hf_max_tokens=4096,
            hf_temperature=1.15,
            log_tag="spiral",
        )
        if not text:
            raise ValueError("All AI lanes failed after full fallback")

        # Belt-and-braces in case the model still wrapped in markdown.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        if not text:
            raise ValueError("empty response after stripping fences")

        # Gemini sometimes emits literal newlines INSIDE string values which
        # is invalid JSON. Try strict parsing first; on failure, attempt a
        # best-effort cleanup before re-parsing.
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            cleaned = _best_effort_json_cleanup(text)
            data = json.loads(cleaned)
        data["_ai_model"] = used_model

        # ---- normalise + validate ----
        outcomes = data.get("outcomes") or []
        if not isinstance(outcomes, list) or len(outcomes) < 1:
            raise ValueError("missing outcomes")
        for o in outcomes:
            o["probability"] = max(0, min(100, int(o.get("probability", 0))))
            sev = str(o.get("severity", "likely")).lower()
            if sev not in {"best", "likely", "worst"}:
                sev = "likely"
            o["severity"] = sev
            o.setdefault("title", "Outcome")
            o.setdefault("description", "")
            o.setdefault("reality_check", "")
        total = sum(o["probability"] for o in outcomes) or 1
        for o in outcomes:
            o["probability"] = round(o["probability"] * 100 / total)
        drift = 100 - sum(o["probability"] for o in outcomes)
        if drift and outcomes:
            outcomes[0]["probability"] += drift
        data["outcomes"] = outcomes

        data.setdefault("in_control", [])
        data.setdefault("out_of_control", [])
        data.setdefault("action_steps", [])

        v = data.get("verdict") or {}
        if isinstance(v, str):
            v = {"verdict_text": v}
        v.setdefault("verdict_text", "Walk off stage.")
        v.setdefault("action_steps", data["action_steps"])
        v.setdefault("in_control", data["in_control"])
        v.setdefault("out_of_control", data["out_of_control"])
        data["verdict"] = v
        data["_ai_source"] = "live"
        print("[gemini] ✅ live response parsed cleanly")
        return data
    except Exception as exc:
        # Loud, structured logging so the user can SEE what failed.
        import traceback
        print(f"[gemini] ❌ FALLBACK — {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return _fallback_with_tag(situation_text, tone, f"{type(exc).__name__}: {str(exc)[:120]}")


# ---------------------------------------------------------------------------
# Compatibility test — analyses two people across multiple
# relationship axes (friends, lovers, FWB, dating, marriage,
# roommates, work partners). Pro-only.
#
# Distinct prompt template from spirals + text-check. The AI is asked
# to commit to scores and verdicts per axis, not hedge. JSON-only.
# ---------------------------------------------------------------------------

class CompatPerson(BaseModel):
    name: str
    gender: Optional[str] = None  # "male" | "female" | "other" | None
    description: Optional[str] = ""
    hobbies: Optional[str] = ""
    status: Optional[str] = ""  # relationship status as free text


class CompatRequest(BaseModel):
    person_a: CompatPerson
    person_b: CompatPerson


def _build_compat_prompt(a: CompatPerson, b: CompatPerson) -> str:
    """Compose the Gemini prompt for a compatibility analysis between
    two people. Asks for SCORES + verdicts per axis with NO hedging."""
    def _block(label: str, p: CompatPerson) -> str:
        bits = [f"Name: {p.name.strip()}"]
        if p.gender: bits.append(f"Gender: {p.gender}")
        if p.status and p.status.strip(): bits.append(f"Current relationship status: {p.status.strip()}")
        if p.description and p.description.strip(): bits.append(f"Description: {p.description.strip()}")
        if p.hobbies and p.hobbies.strip(): bits.append(f"Hobbies / interests: {p.hobbies.strip()}")
        return f"PERSON {label}:\n" + "\n".join(bits)

    return f"""You are a sharp, honest, slightly funny observer of human dynamics. Your job: read two people's profiles and rate how compatible they'd be across SEVEN relationship axes. You commit to scores and verdicts — you do not hedge.

ANTI-PATTERN RULES — read carefully:
- DO NOT say "depends on the chemistry" or "every relationship is unique" or other platitudes.
- DO NOT refuse to score any axis. Pick a number from 0-100 even when uncertain.
- DO NOT moralise about FWB / casual / non-traditional setups. Treat all axes equally.
- DO NOT mention or speculate about race, religion, ethnicity, or sexual orientation unless the profiles literally do.

{_block("A", a)}

{_block("B", b)}

YOUR JOB:
For each of the 7 axes below, return:
  - score (0-100): how WELL these two would do in this specific kind of relationship.
  - verdict (4-8 words): a single committed take.
  - reasoning (1-2 sentences): why. Reference specifics from the profiles when possible.

AXES (return in this exact order):
  1. friends                — platonic friendship
  2. lovers                 — emotional + romantic + sexual partnership
  3. friends_with_benefits  — casual physical with friend energy
  4. dating                 — early-stage romantic, no commitment yet
  5. marriage               — long-haul committed partnership
  6. roommates              — sharing a home, not romantic
  7. work_partners          — co-founding / collaborating professionally

ALSO RETURN:
  - overall_chemistry (0-100): single summary score
  - headline (one sentence, 10-14 words): the punchline of the whole reading
  - green_flags: 2-3 bullets — what specifically WORKS between them
  - red_flags: 2-3 bullets — what specifically DOESN'T

OUTPUT FORMAT — JSON ONLY, NO MARKDOWN:
{{
  "overall_chemistry": <int 0-100>,
  "headline": "<one sentence, 10-14 words>",
  "axes": [
    {{ "axis": "friends",                "score": <int>, "verdict": "...", "reasoning": "..." }},
    {{ "axis": "lovers",                 "score": <int>, "verdict": "...", "reasoning": "..." }},
    {{ "axis": "friends_with_benefits",  "score": <int>, "verdict": "...", "reasoning": "..." }},
    {{ "axis": "dating",                 "score": <int>, "verdict": "...", "reasoning": "..." }},
    {{ "axis": "marriage",               "score": <int>, "verdict": "...", "reasoning": "..." }},
    {{ "axis": "roommates",              "score": <int>, "verdict": "...", "reasoning": "..." }},
    {{ "axis": "work_partners",          "score": <int>, "verdict": "...", "reasoning": "..." }}
  ],
  "green_flags": ["...", "..."],
  "red_flags":   ["...", "..."]
}}

Respond with JSON only.
"""


async def run_compat(a: CompatPerson, b: CompatPerson) -> dict:
    """Call Gemini for the compatibility analysis. Mirrors run_gemini's
    discipline — rotation across GEMINI_MODELS, JSON salvage, structured
    fallback when offline."""
    fallback = {
        "overall_chemistry": 50,
        "headline": "AI is offline — read this as a placeholder, not a verdict.",
        "axes": [
            {"axis": ax, "score": 50, "verdict": "Inconclusive — AI offline.", "reasoning": "Couldn't analyse right now."}
            for ax in ["friends", "lovers", "friends_with_benefits", "dating", "marriage", "roommates", "work_partners"]
        ],
        "green_flags": [], "red_flags": [],
        "_ai_source": "fallback",
    }
    if not GEMINI_API_KEY and not HF_TOKEN:
        return fallback
    try:
        from google.genai import types as gen_types
        config = gen_types.GenerateContentConfig(
            temperature=1.0, top_p=0.95, top_k=40,
            max_output_tokens=2400, response_mime_type="application/json",
        )
        prompt = _build_compat_prompt(a, b)
        # Exhausts every AI lane (Gemini chain → HF DeepSeek chain →
        # full-chain retries) before giving up.
        text, _used_model = await run_with_full_fallback(
            prompt,
            gemini_config=config,
            hf_max_tokens=2400,
            hf_temperature=1.0,
            log_tag="compat",
        )
        if not text:
            raise ValueError("All AI lanes failed after full fallback")
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            data = json.loads(text)
        except Exception:
            data = json.loads(_best_effort_json_cleanup(text))
        # Sanity / clamping
        data["overall_chemistry"] = max(0, min(100, int(data.get("overall_chemistry", 50))))
        axes = data.get("axes") or []
        for ax in axes:
            ax["score"] = max(0, min(100, int(ax.get("score", 50))))
        data["axes"] = axes
        data.setdefault("green_flags", [])
        data.setdefault("red_flags", [])
        data["_ai_source"] = "live"
        return data
    except Exception as exc:
        print(f"[compat] FALLBACK — {type(exc).__name__}: {exc}")
        return {**fallback, "_ai_source": f"fallback: {type(exc).__name__}"}


@app.post("/api/compatibility")
async def compatibility(body: CompatRequest, user: dict = Depends(get_current_user)):
    """Pro-only. Free users get 403 → frontend routes to /paywall.

    Side effect: persists the input + result to compatibility_tests so
    the user can find it in their archive afterwards. saved_id comes
    back for navigation."""
    db_required()
    if user.get("is_guest"):
        raise HTTPException(403, "Sign in to use Compatibility")
    if not _is_pro_tier(user.get("plan_tier")):
        raise HTTPException(403, "Compatibility is a Pro feature. Upgrade for unlimited reads.")
    if not body.person_a.name.strip() or not body.person_b.name.strip():
        raise HTTPException(400, "Both people need a name")
    result = await run_compat(body.person_a, body.person_b)

    # Defensive: make sure the destination table exists in case the
    # lifespan migration didn't run on this deploy.
    await _ensure_archive_tables()

    saved_id = f"cp_{uuid.uuid4().hex[:14]}"

    async def _do_insert():
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO compatibility_tests
                       (id, user_id, person_a, person_b, result, created_at)
                   VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb, NOW())""",
                saved_id, user["user_id"],
                json.dumps(body.person_a.model_dump()),
                json.dumps(body.person_b.model_dump()),
                json.dumps(result),
            )

    persist_error = await _insert_with_ensure(_do_insert, table_label="compat")
    if persist_error is None:
        print(f"[compat] ✅ persisted {saved_id}")
        # Streak counts compatibility tests too — bump the day streak
        # and the activity grid. Fresh user dict for accurate freezes.
        try:
            await bump_activity_streak(user, source="compatibility")
        except Exception as se:
            print(f"[compat] streak bump failed (non-fatal): {se}")
        try:
            await progress_for_event(user, {"kind": "compat_created"})
        except Exception as te:
            print(f"[compat] task progress failed (non-fatal): {te}")
    else:
        saved_id = None

    return {"result": result, "saved_id": saved_id, "persist_error": persist_error}


# ---------------------------------------------------------------------------
# Saved Compatibility CRUD — mirror of the spirals + text-checks CRUD.
# ---------------------------------------------------------------------------

class CompatPatch(BaseModel):
    name: Optional[str] = None
    folder_id: Optional[str] = None
    accent_color: Optional[str] = None
    flagged: Optional[bool] = None


@app.get("/api/compatibilities")
async def list_compatibilities(
    folder_id: Optional[str] = None,
    flagged: Optional[bool] = None,
    limit: int = 200,
    user: dict = Depends(get_current_user),
):
    db_required()
    await _ensure_archive_tables()
    where = ["user_id = $1"]
    args: List[Any] = [user["user_id"]]
    if folder_id == "__unfiled__":
        where.append("folder_id IS NULL")
    elif folder_id:
        args.append(folder_id)
        where.append(f"folder_id = ${len(args)}")
    if flagged is True:
        where.append("flagged = TRUE")
    elif flagged is False:
        where.append("flagged = FALSE")
    args.append(min(max(1, limit), 500))
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT * FROM compatibility_tests
                 WHERE {' AND '.join(where)}
                 ORDER BY created_at DESC
                 LIMIT ${len(args)}""",
            *args,
        )
    return {"compatibilities": [compat_public(dict(r)) for r in rows]}


@app.get("/api/compatibilities/{cp_id}")
async def get_compatibility(cp_id: str, user: dict = Depends(get_current_user)):
    db_required()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM compatibility_tests WHERE id = $1 AND user_id = $2",
            cp_id, user["user_id"],
        )
    if not row:
        raise HTTPException(404, "Compatibility test not found")
    return {"compatibility": compat_public(dict(row))}


@app.patch("/api/compatibilities/{cp_id}")
async def patch_compatibility(cp_id: str, body: CompatPatch, user: dict = Depends(get_current_user)):
    db_required()
    sets, args = [], []
    if body.name is not None:
        sets.append(f"name = ${len(args)+1}"); args.append((body.name or "").strip()[:80] or None)
    if "folder_id" in body.model_fields_set:
        sets.append(f"folder_id = ${len(args)+1}"); args.append(body.folder_id)
    if "accent_color" in body.model_fields_set:
        sets.append(f"accent_color = ${len(args)+1}"); args.append(body.accent_color)
    if body.flagged is not None:
        sets.append(f"flagged = ${len(args)+1}"); args.append(bool(body.flagged))
    if not sets:
        raise HTTPException(400, "Nothing to update")
    args.extend([cp_id, user["user_id"]])
    async with pool.acquire() as conn:
        result = await conn.execute(
            f"UPDATE compatibility_tests SET {', '.join(sets)} WHERE id = ${len(args)-1} AND user_id = ${len(args)}",
            *args,
        )
        if result.endswith("0"):
            raise HTTPException(404, "Compatibility test not found")
        row = await conn.fetchrow("SELECT * FROM compatibility_tests WHERE id = $1", cp_id)
    return {"compatibility": compat_public(dict(row))}


@app.delete("/api/compatibilities/{cp_id}")
async def delete_compatibility(cp_id: str, user: dict = Depends(get_current_user)):
    db_required()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM compatibility_tests WHERE id = $1 AND user_id = $2",
            cp_id, user["user_id"],
        )
        await conn.execute(
            """DELETE FROM item_pairs WHERE user_id = $1
                AND ((a_type = 'compatibility' AND a_id = $2)
                  OR (b_type = 'compatibility' AND b_id = $2))""",
            user["user_id"], cp_id,
        )
    if result.endswith("0"):
        raise HTTPException(404, "Compatibility test not found")
    return {"ok": True}


class CompatSaveRequest(BaseModel):
    person_a: CompatPerson
    person_b: CompatPerson
    result: Dict[str, Any]
    name: Optional[str] = None


@app.post("/api/compatibilities")
async def manual_save_compat(body: CompatSaveRequest, user: dict = Depends(get_current_user)):
    """Manual-save companion to POST /api/compatibility. Accepts a
    pre-computed result so the frontend can retry a failed auto-save
    without re-running the AI."""
    db_required()
    if user.get("is_guest"):
        raise HTTPException(403, "Sign in to save")
    if not _is_pro_tier(user.get("plan_tier")):
        raise HTTPException(403, "Pro required")
    await _ensure_archive_tables()
    if not body.person_a.name.strip() or not body.person_b.name.strip():
        raise HTTPException(400, "Both people need a name")
    saved_id = f"cp_{uuid.uuid4().hex[:14]}"

    async def _do_insert():
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO compatibility_tests
                       (id, user_id, name, person_a, person_b, result, created_at)
                   VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, NOW())""",
                saved_id, user["user_id"], (body.name or None),
                json.dumps(body.person_a.model_dump()),
                json.dumps(body.person_b.model_dump()),
                json.dumps(body.result),
            )

    err = await _insert_with_ensure(_do_insert, table_label="compat-manual")
    if err:
        raise HTTPException(500, f"Save failed: {err}")
    print(f"[compat-manual] ✅ persisted {saved_id}")
    return {"compatibility_id": saved_id}


# ---------------------------------------------------------------------------
# Text-Check ("Don't send that text") — pre-send draft analyzer
#
# Distinct from the spiral pipeline: takes a draft message the user is
# about to send and predicts how the recipient is likely to respond.
# Returns best/likely/worst predicted reply + a verdict (SEND / WAIT /
# REWRITE) + an optional softer rewrite suggestion. Free users get a
# monthly cap; Pro is unlimited.
# ---------------------------------------------------------------------------

# Per-month cap for free users. The full Pro upgrade prompt is in the
# frontend — backend just enforces the wall.
FREE_TEXT_CHECK_MONTHLY = 3


def _build_text_check_prompt(*, draft: str, context: str, relationship: str) -> str:
    """Compose the Gemini prompt for the pre-send analyzer. Returns a
    string ready to send to client.models.generate_content.

    The prompt explicitly forbids platitudes ("trust your gut" etc) and
    asks for tonal specifics — that's what makes the response feel like
    a sharp friend's read instead of a chatbot's hedge.
    """
    rel = relationship.strip() if relationship else "someone"
    ctx_line = f"\nContext from the sender (what's been happening): \"\"\"{context.strip()}\"\"\"" if context.strip() else ""
    return f"""You are reading a draft message someone is about to send to {rel}. Your job: predict three plausible responses (best case, most likely, worst case) and give a one-line VERDICT on whether to send, wait, or rewrite.

ANTI-PATTERN RULES — read carefully:
- DO NOT say "trust your gut" or "go with your heart" or other platitudes.
- DO NOT use the words "anxious", "spiral", "overthinking" in the response.
- DO NOT hedge ("it depends"). Pick a verdict and own it.
- DO NOT moralise about the relationship dynamics.

YOUR JOB:
1. Read the draft below.
2. Predict three plausible responses from the recipient, each grounded in the actual content and tone of the draft.
3. Give a single VERDICT: SEND / WAIT / REWRITE.
4. If verdict is REWRITE, propose a softer/sharper version of the draft (1–2 sentences max).

OUTPUT FORMAT — JSON ONLY, NO MARKDOWN:
{{
  "predicted_responses": [
    {{
      "severity": "best",
      "title": "Short headline (3–6 words). Specific.",
      "reply_text": "A plausible 1–2 sentence response in {rel}'s voice.",
      "probability": <int 0-100>,
      "what_it_means": "One sentence — what this response would tell you."
    }},
    {{ "severity": "likely",  ... }},
    {{ "severity": "worst",   ... }}
  ],
  "verdict": "SEND" | "WAIT" | "REWRITE",
  "verdict_reason": "ONE sentence explaining the verdict. No hedging.",
  "rewrite_suggestion": "<string OR null — only when verdict is REWRITE>",
  "tone_read": "ONE sentence on how the draft will READ to {rel} (e.g. 'reads needy', 'reads cold', 'reads warm but assumes too much')."
}}

HARD RULES:
1. Output VALID JSON ONLY. No prose, no markdown fences.
2. probabilities across the 3 predicted_responses MUST sum to exactly 100.
3. severity values are exactly "best", "likely", "worst" — lowercase.
4. rewrite_suggestion is REQUIRED when verdict == "REWRITE", otherwise null.
5. Be SPECIFIC. Quote phrases from the draft when helpful.

NOW DO IT FOR THIS DRAFT:
Sending to: {rel}{ctx_line}

DRAFT TEXT:
\"\"\"{draft.strip()}\"\"\"

Respond with JSON only.
"""


def _text_check_fallback(reason: str) -> dict:
    """Offline / Gemini-down fallback. Honest about being a placeholder."""
    return {
        "predicted_responses": [
            {"severity": "best",   "title": "Warm reply",            "reply_text": "They respond positively and the conversation moves on.", "probability": 30, "what_it_means": "Best case — no harm done."},
            {"severity": "likely", "title": "Polite acknowledgement", "reply_text": "They say something brief but don't engage with the deeper point.", "probability": 50, "what_it_means": "Most likely — the message lands but doesn't open the conversation you wanted."},
            {"severity": "worst",  "title": "Cool distance",          "reply_text": "They don't reply, or reply curtly. You spend the night re-reading the thread.", "probability": 20, "what_it_means": "Worst case — the silence becomes its own message."},
        ],
        "verdict": "WAIT",
        "verdict_reason": "AI is offline — sleep on it and re-check tomorrow.",
        "rewrite_suggestion": None,
        "tone_read": "Couldn't read the tone — Gemini is unreachable.",
        "_ai_source": f"fallback: {reason}",
    }


async def run_text_check(*, draft: str, context: str, relationship: str) -> dict:
    """Call Gemini with the pre-send prompt. Mirrors run_gemini's
    fallback discipline — model-rotation across GEMINI_MODELS, JSON
    cleanup, loud logging, and an offline-safe fallback."""
    if not GEMINI_API_KEY and not HF_TOKEN:
        print("[text-check] no GEMINI_API_KEY and no HF_TOKEN — using fallback")
        return _text_check_fallback("no_api_keys")
    try:
        from google.genai import types as gen_types
        config = gen_types.GenerateContentConfig(
            temperature=1.05,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )
        prompt = _build_text_check_prompt(draft=draft, context=context, relationship=relationship)
        text, _used_model = await run_with_full_fallback(
            prompt,
            gemini_config=config,
            hf_max_tokens=2048,
            hf_temperature=1.05,
            log_tag="text-check",
        )
        if not text:
            raise ValueError("All AI lanes failed after full fallback")
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            data = json.loads(text)
        except Exception:
            # Reuse the same newline-escape salvage as the spiral path.
            data = json.loads(_best_effort_json_cleanup(text))

        # Normalise + sanity-check the shape.
        responses = data.get("predicted_responses") or []
        if not isinstance(responses, list) or len(responses) != 3:
            raise ValueError("predicted_responses must be a list of 3")
        # Re-normalise probabilities to sum to exactly 100 (Gemini sometimes drifts ±1).
        total = sum(int(r.get("probability", 0)) for r in responses) or 1
        for r in responses:
            r["probability"] = round(int(r.get("probability", 0)) * 100 / total)
        # Round-off correction: shove the residual onto the "likely" bucket.
        drift = 100 - sum(r["probability"] for r in responses)
        if drift != 0:
            likely = next((r for r in responses if r.get("severity") == "likely"), responses[0])
            likely["probability"] = max(0, min(100, likely["probability"] + drift))
        data["predicted_responses"] = responses

        v = (data.get("verdict") or "WAIT").upper()
        if v not in {"SEND", "WAIT", "REWRITE"}:
            v = "WAIT"
        data["verdict"] = v
        data.setdefault("verdict_reason", "Take a beat before sending.")
        data.setdefault("tone_read", "")
        if v != "REWRITE":
            data["rewrite_suggestion"] = None
        else:
            data["rewrite_suggestion"] = data.get("rewrite_suggestion") or None
        data["_ai_source"] = "live"
        return data
    except Exception as exc:
        import traceback
        print(f"[text-check] ❌ FALLBACK — {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return _text_check_fallback(f"{type(exc).__name__}: {str(exc)[:120]}")


class TextCheckRequest(BaseModel):
    draft: str
    context: Optional[str] = ""
    relationship: Optional[str] = "someone"


async def _ensure_text_check_columns():
    """Idempotent: create the text_checks usage-counter columns on
    users so we can enforce FREE_TEXT_CHECK_MONTHLY without a separate
    table. Runs lazily on the first call rather than in lifespan so
    the column add stays close to the feature using it."""
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS text_checks_month TEXT"
        )
        await conn.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS text_checks_count INTEGER DEFAULT 0"
        )


_TEXT_CHECK_COLUMNS_READY = False


def text_check_public(row: dict) -> dict:
    """Shape a text_checks row for the API response. Mirrors
    spiral_public's defensive copy-and-coerce pattern so JSONB columns
    deserialize predictably."""
    out = dict(row)
    # asyncpg returns JSONB as already-decoded dict; if it ever comes
    # back as a string (e.g. older driver), parse defensively.
    res = out.get("result")
    if isinstance(res, str):
        try: out["result"] = json.loads(res)
        except Exception: out["result"] = {}
    if isinstance(out.get("created_at"), datetime):
        out["created_at"] = out["created_at"].isoformat()
    return out


def compat_public(row: dict) -> dict:
    out = dict(row)
    for key in ("result", "person_a", "person_b"):
        v = out.get(key)
        if isinstance(v, str):
            try: out[key] = json.loads(v)
            except Exception: out[key] = {} if key == "result" else {}
    if isinstance(out.get("created_at"), datetime):
        out["created_at"] = out["created_at"].isoformat()
    return out


@app.post("/api/text-check")
async def text_check(body: TextCheckRequest, user: dict = Depends(get_current_user)):
    """Pre-send draft analyzer — Pro-only. Returns three predicted
    responses + a verdict + optional rewrite. Free users get a 403
    with a clear "upgrade" message; the frontend routes that to the
    paywall instead of showing an error.

    Side effect: persists the input + result to the text_checks table
    so the user can find it in their archive afterwards. The saved id
    comes back as `saved_id` for navigation."""
    db_required()
    if user.get("is_guest"):
        raise HTTPException(403, "Sign in to use Text-Check")
    is_pro_user = (user.get("plan_tier") or "free") in {"pro_weekly", "pro_monthly", "pro_yearly", "lifetime"}
    if not is_pro_user:
        # Pro-only. The frontend listens for 403 here and routes to the
        # paywall — the message is what shows in the alert until the
        # paywall opens.
        raise HTTPException(403, "Text-Check is a Pro feature. Upgrade for unlimited drafts.")

    draft = (body.draft or "").strip()
    if not draft:
        raise HTTPException(400, "Draft is empty")
    if len(draft) > 4000:
        raise HTTPException(400, "Draft is too long (4000 char max)")

    result = await run_text_check(
        draft=draft,
        context=body.context or "",
        relationship=body.relationship or "someone",
    )

    # Defensive: make sure the destination table exists in case the
    # lifespan migration didn't run on this deploy.
    await _ensure_archive_tables()

    # Persist — even fallback results are saved so the user keeps the
    # record of their draft. They can delete/rename/folder/pair it.
    saved_id = f"tc_{uuid.uuid4().hex[:14]}"

    async def _do_insert():
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO text_checks
                       (id, user_id, draft, context, relationship, result, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())""",
                saved_id, user["user_id"], draft,
                body.context or "", body.relationship or "someone",
                json.dumps(result),
            )

    persist_error = await _insert_with_ensure(_do_insert, table_label="text-check")
    if persist_error is None:
        print(f"[text-check] ✅ persisted {saved_id}")
        try:
            await bump_activity_streak(user, source="text_check")
        except Exception as se:
            print(f"[text-check] streak bump failed (non-fatal): {se}")
        try:
            await progress_for_event(user, {"kind": "text_check_created"})
        except Exception as te:
            print(f"[text-check] task progress failed (non-fatal): {te}")
    else:
        saved_id = None

    return {
        "result": result,
        "saved_id": saved_id,
        "persist_error": persist_error,
        "usage": {
            "is_pro": True,
            "used_this_month": 0,
            "monthly_limit": None,
            "remaining": None,
        },
    }


# ---------------------------------------------------------------------------
# Saved Text-Checks CRUD — mirror of the spirals archive endpoints.
# All Pro-gated implicitly because the only way to create a row is via
# /api/text-check which is itself Pro-gated.
# ---------------------------------------------------------------------------

class TextCheckPatch(BaseModel):
    name: Optional[str] = None
    folder_id: Optional[str] = None
    accent_color: Optional[str] = None
    flagged: Optional[bool] = None


@app.get("/api/text-checks")
async def list_text_checks(
    folder_id: Optional[str] = None,
    flagged: Optional[bool] = None,
    limit: int = 200,
    user: dict = Depends(get_current_user),
):
    db_required()
    await _ensure_archive_tables()
    where = ["user_id = $1"]
    args: List[Any] = [user["user_id"]]
    if folder_id == "__unfiled__":
        where.append("folder_id IS NULL")
    elif folder_id:
        args.append(folder_id)
        where.append(f"folder_id = ${len(args)}")
    if flagged is True:
        where.append("flagged = TRUE")
    elif flagged is False:
        where.append("flagged = FALSE")
    args.append(min(max(1, limit), 500))
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT * FROM text_checks
                 WHERE {' AND '.join(where)}
                 ORDER BY created_at DESC
                 LIMIT ${len(args)}""",
            *args,
        )
    return {"text_checks": [text_check_public(dict(r)) for r in rows]}


@app.get("/api/text-checks/{tc_id}")
async def get_text_check(tc_id: str, user: dict = Depends(get_current_user)):
    db_required()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM text_checks WHERE id = $1 AND user_id = $2",
            tc_id, user["user_id"],
        )
    if not row:
        raise HTTPException(404, "Text-check not found")
    return {"text_check": text_check_public(dict(row))}


@app.patch("/api/text-checks/{tc_id}")
async def patch_text_check(tc_id: str, body: TextCheckPatch, user: dict = Depends(get_current_user)):
    db_required()
    sets, args = [], []
    if body.name is not None:
        sets.append(f"name = ${len(args)+1}"); args.append((body.name or "").strip()[:80] or None)
    if "folder_id" in body.model_fields_set:
        sets.append(f"folder_id = ${len(args)+1}"); args.append(body.folder_id)
    if "accent_color" in body.model_fields_set:
        sets.append(f"accent_color = ${len(args)+1}"); args.append(body.accent_color)
    if body.flagged is not None:
        sets.append(f"flagged = ${len(args)+1}"); args.append(bool(body.flagged))
    if not sets:
        raise HTTPException(400, "Nothing to update")
    args.extend([tc_id, user["user_id"]])
    async with pool.acquire() as conn:
        result = await conn.execute(
            f"UPDATE text_checks SET {', '.join(sets)} WHERE id = ${len(args)-1} AND user_id = ${len(args)}",
            *args,
        )
        if result.endswith("0"):
            raise HTTPException(404, "Text-check not found")
        row = await conn.fetchrow("SELECT * FROM text_checks WHERE id = $1", tc_id)
    return {"text_check": text_check_public(dict(row))}


@app.delete("/api/text-checks/{tc_id}")
async def delete_text_check(tc_id: str, user: dict = Depends(get_current_user)):
    db_required()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM text_checks WHERE id = $1 AND user_id = $2",
            tc_id, user["user_id"],
        )
        # Cascade-clean any pair rows referencing this item.
        await conn.execute(
            """DELETE FROM item_pairs WHERE user_id = $1
                AND ((a_type = 'text_check' AND a_id = $2)
                  OR (b_type = 'text_check' AND b_id = $2))""",
            user["user_id"], tc_id,
        )
    if result.endswith("0"):
        raise HTTPException(404, "Text-check not found")
    return {"ok": True}


# Manual save — accepts a pre-computed result so the frontend can
# retry persistence after auto-save failed without re-running the AI.
# Different path from POST /api/text-check (which RUNS the AI) so we
# don't accidentally double-bill on retries.
class TextCheckSaveRequest(BaseModel):
    draft: str
    context: Optional[str] = ""
    relationship: Optional[str] = "someone"
    result: Dict[str, Any]
    name: Optional[str] = None


@app.post("/api/text-checks")
async def manual_save_text_check(body: TextCheckSaveRequest, user: dict = Depends(get_current_user)):
    db_required()
    if user.get("is_guest"):
        raise HTTPException(403, "Sign in to save")
    if not _is_pro_tier(user.get("plan_tier")):
        raise HTTPException(403, "Pro required")
    await _ensure_archive_tables()
    draft = (body.draft or "").strip()
    if not draft:
        raise HTTPException(400, "Draft is empty")
    saved_id = f"tc_{uuid.uuid4().hex[:14]}"

    async def _do_insert():
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO text_checks
                       (id, user_id, name, draft, context, relationship, result, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, NOW())""",
                saved_id, user["user_id"], (body.name or None),
                draft, body.context or "", body.relationship or "someone",
                json.dumps(body.result),
            )

    err = await _insert_with_ensure(_do_insert, table_label="text-check-manual")
    if err:
        raise HTTPException(500, f"Save failed: {err}")
    print(f"[text-check-manual] ✅ persisted {saved_id}")
    return {"text_check_id": saved_id}


# ---------------------------------------------------------------------------
# XP / Level / Task system
# ---------------------------------------------------------------------------

def xp_for_level(level: int) -> int:
    """XP needed to reach the next level from the start of this one.
    Curve is steep on purpose — level 100 is the long-haul achievement.
    Total XP to reach L100 ≈ 800,000+."""
    if level < 1:
        return 200
    if level <= 5:
        return 200
    if level <= 10:
        return 500
    if level <= 20:
        return 1200
    if level <= 35:
        return 3000
    if level <= 50:
        return 6500
    if level <= 70:
        return 12000
    if level <= 85:
        return 22000
    return 38000  # 86 → 100 is the grind zone


def total_xp_for_level(level: int) -> int:
    """Cumulative XP required to reach `level` (level 1 = 0 XP)."""
    total = 0
    for lv in range(1, level):
        total += xp_for_level(lv)
    return total


def level_from_xp(xp: int) -> int:
    lv = 1
    while lv < 100 and total_xp_for_level(lv + 1) <= xp:
        lv += 1
    return lv


LEVEL_UNLOCKS = [
    # Every 5 levels = a Thinkpass+ tier reward. Mix of titles, colors, themes.
    (5,   "title",      "The Apprentice",         "Title: The Apprentice"),
    (10,  "name_color", "#C8A0DC",                "Lilac name colour"),
    (15,  "card_theme", "midnight",               "Midnight share-card theme"),
    (20,  "title",      "The Worrier",            "Title: The Worrier"),
    (25,  "name_color", "#F5C518",                "Gold name colour"),
    (30,  "card_theme", "warm",                   "Warm Ember share-card theme"),
    (35,  "title",      "The Analyst",            "Title: The Analyst"),
    (40,  "name_color", "#4FC3F7",                "Ocean name colour"),
    (45,  "card_theme", "forest",                 "Deep Forest share-card theme"),
    (50,  "title",      "Spiral Veteran",         "Title: Spiral Veteran"),
    (55,  "name_color", "#E24B4A",                "Crimson name colour"),
    (60,  "card_theme", "aurora",                 "Aurora Borealis share-card theme"),
    (65,  "title",      "The Philosopher",        "Title: The Philosopher"),
    (70,  "name_color", "#34A56F",                "Moss name colour"),
    (75,  "title",      "The Oracle",             "Title: The Oracle"),
    (80,  "name_color", "#FF8C42",                "Ember name colour"),
    (85,  "card_theme", "neon",                   "Neon Noir share-card theme"),
    (90,  "title",      "Mind Cartographer",      "Title: Mind Cartographer"),
    (95,  "title",      "Overthinking Champion",  "Title: Overthinking Champion"),
    (100, "title",      "Overthinker Supreme",    "Title: Overthinker Supreme"),
]


def unlocks_for_level(level: int) -> list[str]:
    items = []
    for lv, kind, value, _ in LEVEL_UNLOCKS:
        if level >= lv:
            items.append(f"{kind}:{value}")
    return items


DAILY_TASKS = [
    {"task_id": "first_spiral", "label": "Create your first spiral today", "xp": 50, "target": 1},
    {"task_id": "resolve_one", "label": "Resolve one spiral", "xp": 75, "target": 1},
    {"task_id": "brutal_tone", "label": "Use the Brutal tone", "xp": 40, "target": 1},
    {"task_id": "long_spiral", "label": "Write a 100+ word spiral", "xp": 60, "target": 1},
    {"task_id": "share_verdict", "label": "Share a verdict", "xp": 40, "target": 1},
    {"task_id": "plot_twist", "label": "Log a plot-twist resolution", "xp": 80, "target": 1},
    # Pro feature tasks — count text-check and compatibility usage as
    # daily wins so the gamification reaches every part of the app.
    {"task_id": "one_text_check", "label": "Stop one text from going out", "xp": 60, "target": 1},
    {"task_id": "one_compat", "label": "Run a compatibility test", "xp": 60, "target": 1},
]

WEEKLY_TASKS = [
    {"task_id": "seven_spirals", "label": "Create 7 spirals this week", "xp": 500, "target": 7},
    {"task_id": "five_resolved", "label": "Resolve 5 spirals", "xp": 400, "target": 5},
    {"task_id": "all_tones", "label": "Use all 3 tones", "xp": 300, "target": 3},
    {"task_id": "five_streak", "label": "Hit a 5-day streak", "xp": 350, "target": 5},
    {"task_id": "four_categories", "label": "4 different categories", "xp": 450, "target": 4},
    {"task_id": "share_three", "label": "Share 3 verdicts", "xp": 250, "target": 3},
    {"task_id": "gentle_three", "label": "Use the Gentle tone 3x", "xp": 180, "target": 3},
    {"task_id": "three_streak", "label": "Hit a 3-day streak", "xp": 200, "target": 3},
    # Cross-feature weekly goals.
    {"task_id": "three_text_checks", "label": "Run 3 text-checks", "xp": 280, "target": 3},
    {"task_id": "two_compats", "label": "Run 2 compatibility tests", "xp": 240, "target": 2},
    {"task_id": "all_three_kinds", "label": "Use all 3 tools (spiral / text / compat)", "xp": 350, "target": 3},
]


def current_daily_period() -> str:
    return today_iso_date()


def current_weekly_period() -> str:
    today = datetime.now(timezone.utc).date()
    monday = today - timedelta(days=today.weekday())
    return f"W{monday.isoformat()}"


async def ensure_task_rows(user_id: str) -> None:
    day = current_daily_period()
    week = current_weekly_period()
    async with pool.acquire() as conn:
        for t in DAILY_TASKS:
            await conn.execute(
                """INSERT INTO user_tasks (user_id, task_id, period, target, progress, created_at)
                   VALUES ($1,$2,$3,$4,0,$5)
                   ON CONFLICT (user_id, task_id, period) DO NOTHING""",
                user_id, t["task_id"], day, t["target"], now_iso(),
            )
        for t in WEEKLY_TASKS:
            await conn.execute(
                """INSERT INTO user_tasks (user_id, task_id, period, target, progress, created_at)
                   VALUES ($1,$2,$3,$4,0,$5)
                   ON CONFLICT (user_id, task_id, period) DO NOTHING""",
                user_id, t["task_id"], week, t["target"], now_iso(),
            )


async def bump_task(user_id: str, task_id: str, period: str, by: int = 1) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE user_tasks SET progress = LEAST(progress + $1, target) "
            "WHERE user_id = $2 AND task_id = $3 AND period = $4",
            by, user_id, task_id, period,
        )


async def progress_for_event(user: dict, event: dict) -> None:
    """Apply task progress for various events."""
    if user.get("is_guest") or (user.get("plan_tier") or "free") == "free":
        return
    await ensure_task_rows(user["user_id"])
    day = current_daily_period()
    week = current_weekly_period()

    kind = event.get("kind")
    if kind == "spiral_created":
        await bump_task(user["user_id"], "first_spiral", day)
        await bump_task(user["user_id"], "seven_spirals", week)
        await _bump_all_three_kinds(user["user_id"], week, "spiral")
        tone = event.get("tone")
        if tone == "brutal":
            await bump_task(user["user_id"], "brutal_tone", day)
        if tone == "gentle":
            await bump_task(user["user_id"], "gentle_three", week)
        if event.get("words", 0) >= 100:
            await bump_task(user["user_id"], "long_spiral", day)
        # all-tones / categories tracked via stored lists
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT tones_used, categories_used FROM user_tasks "
                "WHERE user_id = $1 AND task_id = 'all_tones' AND period = $2",
                user["user_id"], week,
            )
            tones = list(row["tones_used"] or []) if row else []
            if isinstance(tones, str):
                try:
                    tones = json.loads(tones)
                except Exception:
                    tones = []
            if tone and tone not in tones:
                tones.append(tone)
                await conn.execute(
                    "UPDATE user_tasks SET tones_used = $1, progress = LEAST($2, target) "
                    "WHERE user_id = $3 AND task_id = 'all_tones' AND period = $4",
                    json.dumps(tones), len(tones), user["user_id"], week,
                )

            cat = event.get("category")
            row = await conn.fetchrow(
                "SELECT categories_used FROM user_tasks "
                "WHERE user_id = $1 AND task_id = 'four_categories' AND period = $2",
                user["user_id"], week,
            )
            cats = list(row["categories_used"] or []) if row else []
            if isinstance(cats, str):
                try:
                    cats = json.loads(cats)
                except Exception:
                    cats = []
            if cat and cat not in cats:
                cats.append(cat)
                await conn.execute(
                    "UPDATE user_tasks SET categories_used = $1, progress = LEAST($2, target) "
                    "WHERE user_id = $3 AND task_id = 'four_categories' AND period = $4",
                    json.dumps(cats), len(cats), user["user_id"], week,
                )

    elif kind == "spiral_resolved":
        await bump_task(user["user_id"], "resolve_one", day)
        await bump_task(user["user_id"], "five_resolved", week)
        if event.get("status") == "other":
            await bump_task(user["user_id"], "plot_twist", day)

    elif kind == "share":
        await bump_task(user["user_id"], "share_verdict", day)
        await bump_task(user["user_id"], "share_three", week)

    elif kind == "text_check_created":
        await bump_task(user["user_id"], "one_text_check", day)
        await bump_task(user["user_id"], "three_text_checks", week)
        await _bump_all_three_kinds(user["user_id"], week, "text_check")

    elif kind == "compat_created":
        await bump_task(user["user_id"], "one_compat", day)
        await bump_task(user["user_id"], "two_compats", week)
        await _bump_all_three_kinds(user["user_id"], week, "compat")

    # Cross-task: the spiral_created branch above also needs to record
    # the "spiral" kind for the all-three weekly. Inlined there.


async def _bump_all_three_kinds(user_id: str, week: str, kind: str) -> None:
    """Track which of {spiral, text_check, compat} the user has used
    this week. Stored as a JSON list on the task row so we can detect
    "saw all three" without a separate table."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT tones_used FROM user_tasks WHERE user_id = $1 AND task_id = 'all_three_kinds' AND period = $2",
            user_id, week,
        )
        seen = list(row["tones_used"] or []) if row else []
        if isinstance(seen, str):
            try: seen = json.loads(seen)
            except Exception: seen = []
        if kind not in seen:
            seen.append(kind)
            await conn.execute(
                "UPDATE user_tasks SET tones_used = $1, progress = LEAST($2, target) "
                "WHERE user_id = $3 AND task_id = 'all_three_kinds' AND period = $4",
                json.dumps(seen), len(seen), user_id, week,
            )


# ---------------------------------------------------------------------------
# Streak handling
# ---------------------------------------------------------------------------

def _date_diff_days(a_iso: Optional[str], b_iso: Optional[str]) -> Optional[int]:
    """Return |a - b| in days when both are YYYY-MM-DD strings, else None."""
    if not a_iso or not b_iso:
        return None
    try:
        a = datetime.fromisoformat(a_iso).date() if "T" not in a_iso else datetime.fromisoformat(a_iso.replace("Z", "+00:00")).date()
        b = datetime.fromisoformat(b_iso).date() if "T" not in b_iso else datetime.fromisoformat(b_iso.replace("Z", "+00:00")).date()
        return abs((a - b).days)
    except Exception:
        return None


def _streak_freeze_state(user: dict) -> tuple[int, str]:
    """Resolve the user's current per-month freeze allotment, regenerating
    if the calendar month rolled over since the last write. Returns
    (remaining, month). Does not persist — callers are expected to write
    them back through whatever UPDATE they're already issuing."""
    this_month = datetime.now(timezone.utc).strftime("%Y-%m")
    stored_month = user.get("streak_freezes_month")
    if stored_month != this_month:
        # New month → reset to full allotment (3). Free users never get
        # to USE these but the column existing for everyone keeps the
        # schema simple; the burn logic is what gates Pro-only access.
        return (3, this_month)
    remaining = user.get("streak_freezes_remaining")
    if remaining is None:
        remaining = 3
    return (max(0, int(remaining)), this_month)


def _is_pro_tier(plan_tier: Optional[str]) -> bool:
    return (plan_tier or "free") in {"pro_weekly", "pro_monthly", "pro_yearly", "lifetime"}


async def _record_streak_activity(
    conn,
    user_id: str,
    today_iso: str,
    source: str,
    freeze_dates: List[str],
) -> None:
    """Upsert today's activity row + record any freeze-rescued dates.
    `source` is one of 'spiral' | 'text_check' | 'compatibility' — drives
    which per-type counter increments. Designed to be called from inside
    an existing conn block to avoid extra pool acquires."""
    col = {
        "spiral": "spiral_count",
        "text_check": "text_check_count",
        "compatibility": "compat_count",
    }.get(source, "spiral_count")
    try:
        await conn.execute(
            f"""INSERT INTO streak_activity (user_id, activity_date, kind, {col})
                  VALUES ($1, $2::date, 'active', 1)
                ON CONFLICT (user_id, activity_date) DO UPDATE
                  SET {col} = streak_activity.{col} + 1,
                      kind = CASE WHEN streak_activity.kind = 'freeze' THEN 'active' ELSE streak_activity.kind END""",
            user_id, today_iso,
        )
        for d in freeze_dates:
            await conn.execute(
                """INSERT INTO streak_activity (user_id, activity_date, kind)
                       VALUES ($1, $2::date, 'freeze')
                       ON CONFLICT (user_id, activity_date) DO NOTHING""",
                user_id, d,
            )
    except Exception as exc:
        # Non-fatal — log and move on so a streak_activity outage can't
        # block the actual save.
        print(f"[streak_activity] write failed: {type(exc).__name__}: {exc}")


async def bump_activity_streak(user: dict, *, source: str) -> None:
    """Generic day-streak bumper. Called from text-check and
    compatibility endpoints (and indirectly by spiral via the
    bump_streak_and_total path). Doesn't touch spirals_used_today /
    spirals_total — those are spiral-specific and stay in
    bump_streak_and_total. Records streak_activity for the heatmap."""
    today = today_iso_date()
    last = user.get("last_spiral_date")
    streak = user.get("streak_count") or 0
    diff = _date_diff_days(last, today)

    freezes_remaining, freeze_month = _streak_freeze_state(user)
    pro = _is_pro_tier(user.get("plan_tier"))
    freeze_dates: List[str] = []

    if diff == 0:
        new_streak = streak or 1
    elif diff == 1:
        new_streak = streak + 1
    elif diff is not None and 2 <= diff <= 4 and pro and freezes_remaining >= (diff - 1):
        # Bridge — record each rescued date so the heatmap shows the
        # freeze tile for that day.
        try:
            today_date = datetime.fromisoformat(today).date()
            for d in range(1, diff):
                freeze_dates.append((today_date - timedelta(days=d)).isoformat())
        except Exception:
            pass
        freezes_remaining -= max(0, diff - 1)
        new_streak = streak + 1
    else:
        new_streak = 1

    async with pool.acquire() as conn:
        await conn.execute(
            """UPDATE users SET
                  streak_count = $1,
                  last_active = $2,
                  last_spiral_date = $3,
                  streak_freezes_remaining = $4,
                  streak_freezes_month = $5
                WHERE user_id = $6""",
            new_streak, now_iso(), today,
            freezes_remaining, freeze_month, user["user_id"],
        )
        await _record_streak_activity(conn, user["user_id"], today, source, freeze_dates)
    if freeze_dates:
        print(f"[streak] burned {len(freeze_dates)} freeze(s) for {user['user_id']} via {source}")


async def bump_streak_and_total(user: dict) -> dict:
    """Called on every successful spiral creation. Bumps the day counter,
    advances the streak if it's consecutive, and burns Pro freezes to
    bridge 1–3 day gaps when possible.

    Rules:
      • diff == 0 → already spiralled today, no streak change.
      • diff == 1 → consecutive day, streak += 1.
      • 2 <= diff <= 4 AND Pro AND enough freezes → burn (diff - 1)
        freezes to bridge the gap, streak += 1.
      • Otherwise → streak resets to 1.

    diff is measured against last_spiral_date (not last_active — the
    latter also moves on Google sign-in, which would forgive missed
    days for free.)
    """
    today = today_iso_date()
    last_spiral = user.get("last_spiral_date")
    streak = user.get("streak_count") or 0
    diff = _date_diff_days(last_spiral, today)

    freezes_remaining, freeze_month = _streak_freeze_state(user)
    pro = _is_pro_tier(user.get("plan_tier"))
    freezes_burned = 0
    freeze_dates: List[str] = []

    if diff == 0:
        new_streak = streak or 1
    elif diff == 1:
        new_streak = streak + 1
    elif diff is not None and 2 <= diff <= 4 and pro and freezes_remaining >= (diff - 1):
        # Bridge the gap with freezes — one freeze per missed day.
        freezes_burned = diff - 1
        freezes_remaining -= freezes_burned
        new_streak = streak + 1
        try:
            today_date = datetime.fromisoformat(today).date()
            for d in range(1, diff):
                freeze_dates.append((today_date - timedelta(days=d)).isoformat())
        except Exception:
            pass
    else:
        new_streak = 1

    used_today_date = user.get("spirals_used_date")
    if used_today_date == today:
        used_today = (user.get("spirals_used_today") or 0) + 1
    else:
        used_today = 1

    async with pool.acquire() as conn:
        await conn.execute(
            """UPDATE users SET
                streak_count = $1,
                last_active = $2,
                last_spiral_date = $3,
                spirals_used_today = $4,
                spirals_used_date = $5,
                spirals_total = spirals_total + 1,
                streak_freezes_remaining = $6,
                streak_freezes_month = $7
               WHERE user_id = $8""",
            new_streak, now_iso(), today, used_today, today,
            freezes_remaining, freeze_month, user["user_id"],
        )
        if freezes_burned > 0:
            print(f"[streak] burned {freezes_burned} freeze(s) for user {user['user_id']} (gap {diff} days)")
        # Record this spiral in the streak heatmap (active today + any
        # freeze-rescued dates).
        await _record_streak_activity(conn, user["user_id"], today, "spiral", freeze_dates)
        return dict(await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user["user_id"]))


def _compute_streak_view(user: dict) -> dict:
    """Read-side streak state. Recomputes on every /api/auth/me so a
    user who hasn't opened the app for days sees the correct "broken"
    state without us needing a background job.

    Returns a small dict the frontend can render directly:
      {
        "streak_count": int,           # zero if effectively broken
        "freezes_remaining": int,      # always present, gated by is_pro at UI level
        "freezes_max": 3,
        "is_pro": bool,
        "days_since_last_spiral": int | None,
        "in_danger": bool,             # 1-2 days since last spiral, not broken yet
        "broken_today": bool,          # streak just lost today (>4 days gap or out of freezes)
        "last_spiral_date": str | None,
      }
    """
    today = today_iso_date()
    last_spiral = user.get("last_spiral_date")
    diff = _date_diff_days(last_spiral, today)
    streak = user.get("streak_count") or 0
    pro = _is_pro_tier(user.get("plan_tier"))
    freezes_remaining, _ = _streak_freeze_state(user)

    in_danger = False
    broken_today = False
    effective_streak = streak

    if diff is not None and streak > 0:
        # diff == 0 → spiralled today, streak fresh.
        # diff == 1 → in danger (will reset to 1 unless they spiral today).
        # diff >= 2 → in danger; if they have freezes they're safe.
        # diff > 4 → broken even with freezes.
        if diff == 1:
            in_danger = True
        elif 2 <= diff <= 4:
            if pro and freezes_remaining >= (diff - 1):
                in_danger = True  # safe but on the edge
            else:
                broken_today = True
                effective_streak = 0
        elif diff > 4:
            broken_today = True
            effective_streak = 0

    return {
        "streak_count": effective_streak,
        "freezes_remaining": freezes_remaining,
        "freezes_max": 3,
        "is_pro": pro,
        "days_since_last_spiral": diff,
        "in_danger": in_danger,
        "broken_today": broken_today,
        "last_spiral_date": last_spiral,
    }


async def grant_xp(user_id: str, amount: int) -> dict:
    """Adds XP and bumps level. Does NOT auto-grant cosmetics anymore —
    the user must claim each tier from Thinkpass+ explicitly."""
    async with pool.acquire() as conn:
        u = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user_id)
        if not u:
            return {}
        new_xp = (u["xp"] or 0) + amount
        new_level = level_from_xp(new_xp)
        await conn.execute(
            "UPDATE users SET xp = $1, level = $2 WHERE user_id = $3",
            new_xp, new_level, user_id,
        )
        return {"xp": new_xp, "level": new_level}


# ---------------------------------------------------------------------------
# Spirals
# ---------------------------------------------------------------------------

@app.post("/api/spirals")
async def create_spiral(body: SpiralCreate, user: dict = Depends(get_current_user)):
    db_required()
    # Paywall: free tier capped at lifetime total
    if (user.get("plan_tier") or "free") == "free" and (user.get("spirals_total") or 0) >= FREE_LIFETIME_LIMIT:
        raise HTTPException(402, "Free tier limit reached. Upgrade to keep going.")

    # Themed drops (#7) are Pro-only. Also reject the request if the
    # drop window has closed between when the frontend last loaded
    # /api/drops/current and now (defensive — UI should hide it once
    # ended, but a stale client could still try).
    if body.tone.startswith("drop:"):
        if not _is_pro_tier(user.get("plan_tier")):
            raise HTTPException(403, "Themed drops are a Pro feature. Upgrade to use them.")
        drop = _drop_by_id(body.tone.split(":", 1)[1])
        today = today_iso_date()
        if not drop or not (drop["start"] <= today < drop["end"]):
            raise HTTPException(400, "That drop is no longer active.")

    spiral_id = f"sp_{uuid.uuid4().hex[:18]}"
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO spirals (id, user_id, situation_text, category, tone_used, status, created_at)
               VALUES ($1,$2,$3,$4,$5,'processing',$6)""",
            spiral_id, user["user_id"], body.situation_text, body.category, body.tone, now_iso(),
        )

    # Kick off Gemini in the background so the HTTP response returns
    # immediately. The client then navigates to the processing screen
    # which polls until status='complete'. This makes the loading animation
    # actually visible (instead of the user staring at a spinner on the
    # input screen while Gemini works).
    import asyncio as _asyncio
    _asyncio.create_task(_run_spiral_generation(
        spiral_id,
        body.situation_text,
        body.tone,
        body.category,
        user,
    ))

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM spirals WHERE id = $1", spiral_id)
    return {"spiral": spiral_public(dict(row))}


async def _run_spiral_generation(
    spiral_id: str,
    situation_text: str,
    tone: str,
    category: str,
    user: dict,
) -> None:
    """Background task — runs Gemini, persists result, updates XP/streak.
    Errors are swallowed so they don't take down the request loop."""
    try:
        payload = await run_gemini(situation_text, tone, category)
        ai_source = payload.get("_ai_source", "unknown")
        ai_reason = payload.get("_ai_reason")

        # Soundtrack — only persist if Gemini returned a well-shaped
        # object. Fallback paths skip it (the field is optional and the
        # frontend renders nothing if it's null).
        soundtrack_obj = payload.get("soundtrack")
        soundtrack_json = None
        if isinstance(soundtrack_obj, dict):
            t = (soundtrack_obj.get("title") or "").strip()
            l1 = (soundtrack_obj.get("line_1") or "").strip()
            l2 = (soundtrack_obj.get("line_2") or "").strip()
            if t and l1 and l2:
                soundtrack_json = json.dumps({
                    "title": t[:60], "line_1": l1[:120], "line_2": l2[:120],
                })

        async with pool.acquire() as conn:
            await conn.execute(
                """UPDATE spirals SET
                    outcomes = $1,
                    verdict = $2,
                    status = 'complete',
                    error_message = $3,
                    name = COALESCE(name, $5),
                    soundtrack = $6
                   WHERE id = $4""",
                json.dumps(payload.get("outcomes", [])),
                json.dumps(payload.get("verdict", {})),
                f"fallback: {ai_reason}" if ai_source == "fallback" else "live",
                spiral_id,
                (payload.get("name") or "Untitled").strip()[:32],
                soundtrack_json,
            )

        updated_user = await bump_streak_and_total(user)
        await progress_for_event(updated_user, {
            "kind": "spiral_created",
            "tone": tone,
            "category": category,
            "words": len((situation_text or "").split()),
        })
        await grant_xp(updated_user["user_id"], 10)
    except Exception as exc:
        print(f"[spiral_gen] background task failed for {spiral_id}: {exc!r}")
        # Mark spiral as errored so the polling client doesn't spin forever
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE spirals SET status = 'error', error_message = $1 WHERE id = $2",
                    f"generation_failed: {type(exc).__name__}",
                    spiral_id,
                )
        except Exception:
            pass


@app.get("/api/spirals")
async def list_spirals(
    user: dict = Depends(get_current_user),
    category: Optional[str] = None,
    archive: bool = False,
    resolved: Optional[bool] = None,
    limit: int = 50,
):
    limit = max(1, min(limit, 200))
    sql = "SELECT * FROM spirals WHERE user_id = $1"
    args: list[Any] = [user["user_id"]]
    if category:
        args.append(category)
        sql += f" AND category = ${len(args)}"
    if resolved is not None:
        args.append(resolved)
        sql += f" AND resolved = ${len(args)}"
    sql += " ORDER BY created_at DESC"
    args.append(limit)
    sql += f" LIMIT ${len(args)}"
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return {"spirals": [spiral_public(dict(r)) for r in rows]}


@app.get("/api/spirals/stats")
async def spirals_stats(user: dict = Depends(get_current_user)):
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM spirals WHERE user_id = $1", user["user_id"])
        resolved = await conn.fetchval(
            "SELECT COUNT(*) FROM spirals WHERE user_id = $1 AND resolved = TRUE",
            user["user_id"],
        )
    total = total or 0
    resolved = resolved or 0
    pct = round((resolved / total) * 100) if total else 0
    return {"total": total, "resolved": resolved, "resolved_pct": pct}


@app.get("/api/spirals/{spiral_id}")
async def get_spiral(spiral_id: str, user: dict = Depends(get_current_user)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM spirals WHERE id = $1 AND user_id = $2",
            spiral_id, user["user_id"],
        )
    if not row:
        raise HTTPException(404, "Spiral not found")
    spiral = spiral_public(dict(row))
    # Pattern Alerts (#5) + Spiral Soundtrack (#10) — both Pro-only.
    # Strip / skip for free users so the frontend has nothing to render.
    is_pro_user = (user.get("plan_tier") or "free") in {"pro_weekly", "pro_monthly", "pro_yearly", "lifetime"}
    if is_pro_user:
        spiral["pattern_context"] = await _compute_pattern_context(
            user_id=user["user_id"],
            current_spiral=dict(row),
        )
    else:
        spiral["pattern_context"] = None
        spiral["soundtrack"] = None
    return {"spiral": spiral}


async def _compute_pattern_context(*, user_id: str, current_spiral: dict) -> Optional[dict]:
    """Find past spirals that match THIS one's pattern (same category +
    Jaccard tag overlap ≥ 0.34). Returns:

        {
          "count": int,                       # incl. this spiral
          "last_spiral_id": str | None,       # most recent prior match
          "last_resolution_status": str | None,
          "last_resolution_note": str | None,
          "last_resolution_at": str | None,
        }

    Or None when there's no meaningful pattern (count < 2). The frontend
    only renders the banner when this returns a dict — never when None.
    """
    cat = current_spiral.get("category") or "other"
    cur_tags = _tag_key(current_spiral.get("tags"))
    cur_id = current_spiral.get("id")
    cur_created = current_spiral.get("created_at")
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, tags, resolution_status, resolution_note,
                      resolved_at, created_at
               FROM spirals
               WHERE user_id = $1
                 AND status = 'complete'
                 AND category = $2
                 AND id <> $3
               ORDER BY created_at DESC""",
            user_id, cat, cur_id,
        )
    matches = []
    for r in rows:
        rd = dict(r)
        if _jaccard(_tag_key(rd.get("tags")), cur_tags) >= 0.34 or (
            # Empty-tag spirals only match other empty-tag spirals in
            # the same category — same rule as the clustering pass.
            not cur_tags and not _tag_key(rd.get("tags"))
        ):
            matches.append(rd)
    count_incl = len(matches) + 1
    if count_incl < 2:
        return None
    # Most recent past match (already sorted DESC). Prefer one that has
    # a resolution so the banner copy can show "last one resolved as X".
    last_resolved = next(
        (m for m in matches if m.get("resolution_status")), matches[0]
    )
    def _to_iso(v):
        if v is None:
            return None
        return v.isoformat() if hasattr(v, "isoformat") else str(v)
    return {
        "count": count_incl,
        "last_spiral_id": last_resolved.get("id"),
        "last_resolution_status": last_resolved.get("resolution_status"),
        "last_resolution_note": last_resolved.get("resolution_note"),
        "last_resolution_at": _to_iso(last_resolved.get("resolved_at")),
    }


@app.patch("/api/spirals/{spiral_id}/resolve")
async def resolve_spiral(spiral_id: str, body: SpiralResolve, user: dict = Depends(get_current_user)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM spirals WHERE id = $1 AND user_id = $2",
            spiral_id, user["user_id"],
        )
        if not row:
            raise HTTPException(404, "Spiral not found")
        await conn.execute(
            """UPDATE spirals SET
                resolved = $1,
                resolution_status = $2,
                resolution_note = $3,
                resolved_at = $4
               WHERE id = $5""",
            body.resolved,
            body.resolution_status,
            body.resolution_note,
            now_iso() if body.resolved else None,
            spiral_id,
        )
        row = await conn.fetchrow("SELECT * FROM spirals WHERE id = $1", spiral_id)
    if body.resolved:
        await progress_for_event(user, {"kind": "spiral_resolved", "status": body.resolution_status})
        await grant_xp(user["user_id"], 25)
    return {"spiral": spiral_public(dict(row))}


@app.patch("/api/spirals/{spiral_id}")
async def patch_spiral(spiral_id: str, body: SpiralPatch, user: dict = Depends(get_current_user)):
    sets, args = [], []
    if body.category is not None:
        sets.append(f"category = ${len(args)+1}")
        args.append(body.category)
    if body.tags is not None:
        sets.append(f"tags = ${len(args)+1}")
        args.append(json.dumps(body.tags))
    if body.flagged is not None:
        sets.append(f"flagged = ${len(args)+1}")
        args.append(body.flagged)
    # folder_id support — we accept the field whenever it was explicitly
    # present in the JSON, including when it was set to null (which means
    # "remove from folder"). model_fields_set works for pydantic v2; if it
    # fails we treat any presence as intent to update.
    try:
        folder_id_in_body = "folder_id" in body.model_fields_set
    except Exception:
        folder_id_in_body = body.folder_id is not None
    if folder_id_in_body:
        sets.append(f"folder_id = ${len(args)+1}")
        args.append(body.folder_id)
    # accent_color uses the same explicit-null-allowed pattern
    try:
        accent_in_body = "accent_color" in body.model_fields_set
    except Exception:
        accent_in_body = body.accent_color is not None
    if accent_in_body:
        sets.append(f"accent_color = ${len(args)+1}")
        args.append(body.accent_color)
    if body.name is not None:
        sets.append(f"name = ${len(args)+1}")
        args.append(body.name.strip()[:32] or None)
    if not sets:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM spirals WHERE id = $1 AND user_id = $2",
                spiral_id, user["user_id"],
            )
        if not row:
            raise HTTPException(404, "Spiral not found")
        return {"spiral": spiral_public(dict(row))}
    args.extend([spiral_id, user["user_id"]])
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE spirals SET {', '.join(sets)} WHERE id = ${len(args)-1} AND user_id = ${len(args)}",
            *args,
        )
        row = await conn.fetchrow(
            "SELECT * FROM spirals WHERE id = $1 AND user_id = $2",
            spiral_id, user["user_id"],
        )
    if not row:
        raise HTTPException(404, "Spiral not found")
    return {"spiral": spiral_public(dict(row))}


# ---------------------------------------------------------------------------
# Folders — simple flat list per user. Each spiral can belong to one folder.
# ---------------------------------------------------------------------------

@app.get("/api/folders")
async def list_folders(user: dict = Depends(get_current_user)):
    db_required()
    # item_count aggregates across all three item kinds (spirals,
    # text-checks, compatibility tests) since the user can drop any of
    # them into a folder. spiral_count is kept for backwards-compat
    # with older frontends that still read that field.
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT f.id, f.name, f.color, f.created_at,
                      (SELECT COUNT(*) FROM spirals s
                         WHERE s.folder_id = f.id AND s.user_id = f.user_id) AS spiral_count,
                      (SELECT COUNT(*) FROM text_checks tc
                         WHERE tc.folder_id = f.id AND tc.user_id = f.user_id) AS text_check_count,
                      (SELECT COUNT(*) FROM compatibility_tests ct
                         WHERE ct.folder_id = f.id AND ct.user_id = f.user_id) AS compat_count
                 FROM folders f
                WHERE f.user_id = $1
                ORDER BY f.created_at DESC""",
            user["user_id"],
        )
    out = []
    for r in rows:
        d = dict(r)
        d["item_count"] = int(d["spiral_count"] or 0) + int(d["text_check_count"] or 0) + int(d["compat_count"] or 0)
        out.append(d)
    return {"folders": out}


@app.post("/api/folders")
async def create_folder(body: FolderCreate, user: dict = Depends(get_current_user)):
    db_required()
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(400, "Folder name required.")
    folder_id = f"fd_{uuid.uuid4().hex[:14]}"
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO folders (id, user_id, name, color, created_at)
                 VALUES ($1, $2, $3, $4, $5)""",
            folder_id, user["user_id"], name[:80], body.color, now_iso(),
        )
        row = await conn.fetchrow("SELECT * FROM folders WHERE id = $1", folder_id)
    return {"folder": dict(row)}


@app.patch("/api/folders/{folder_id}")
async def rename_folder(folder_id: str, body: FolderRename, user: dict = Depends(get_current_user)):
    db_required()
    sets, args = [], []
    if body.name is not None:
        n = body.name.strip()
        if not n:
            raise HTTPException(400, "Folder name cannot be empty.")
        sets.append(f"name = ${len(args)+1}")
        args.append(n[:80])
    try:
        color_in_body = "color" in body.model_fields_set
    except Exception:
        color_in_body = body.color is not None
    if color_in_body:
        sets.append(f"color = ${len(args)+1}")
        args.append(body.color)
    if not sets:
        raise HTTPException(400, "Nothing to update.")
    args.extend([folder_id, user["user_id"]])
    async with pool.acquire() as conn:
        result = await conn.execute(
            f"UPDATE folders SET {', '.join(sets)} WHERE id = ${len(args)-1} AND user_id = ${len(args)}",
            *args,
        )
        if result.endswith("0"):
            raise HTTPException(404, "Folder not found")
        row = await conn.fetchrow("SELECT * FROM folders WHERE id = $1", folder_id)
    return {"folder": dict(row)}


@app.delete("/api/folders/{folder_id}")
async def delete_folder(folder_id: str, user: dict = Depends(get_current_user)):
    db_required()
    async with pool.acquire() as conn:
        # Detach all three item kinds from the deleted folder (don't
        # delete the items themselves — they move to Unfiled).
        await conn.execute(
            "UPDATE spirals SET folder_id = NULL WHERE folder_id = $1 AND user_id = $2",
            folder_id, user["user_id"],
        )
        await conn.execute(
            "UPDATE text_checks SET folder_id = NULL WHERE folder_id = $1 AND user_id = $2",
            folder_id, user["user_id"],
        )
        await conn.execute(
            "UPDATE compatibility_tests SET folder_id = NULL WHERE folder_id = $1 AND user_id = $2",
            folder_id, user["user_id"],
        )
        await conn.execute(
            "DELETE FROM folders WHERE id = $1 AND user_id = $2",
            folder_id, user["user_id"],
        )
    return {"ok": True}


@app.post("/api/spirals/{spiral_id}/share")
async def share_spiral(spiral_id: str, user: dict = Depends(get_current_user)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM spirals WHERE id = $1 AND user_id = $2",
            spiral_id, user["user_id"],
        )
        if not row:
            raise HTTPException(404, "Spiral not found")
        await conn.execute(
            "UPDATE spirals SET share_count = share_count + 1 WHERE id = $1",
            spiral_id,
        )
        await conn.execute(
            "UPDATE users SET shared_count = shared_count + 1 WHERE user_id = $1",
            user["user_id"],
        )
    await progress_for_event(user, {"kind": "share"})
    return {"ok": True}


@app.delete("/api/spirals/{spiral_id}")
async def delete_spiral(spiral_id: str, user: dict = Depends(get_current_user)):
    async with pool.acquire() as conn:
        res = await conn.execute(
            "DELETE FROM spirals WHERE id = $1 AND user_id = $2",
            spiral_id, user["user_id"],
        )
        if "DELETE 0" in res:
            raise HTTPException(404, "Spiral not found")
        await conn.execute(
            "UPDATE users SET deleted_count = deleted_count + 1 WHERE user_id = $1",
            user["user_id"],
        )
        # Drop any cross-type pair rows that referenced this spiral so
        # the archive's "related" lists don't show ghost links.
        await conn.execute(
            """DELETE FROM item_pairs WHERE user_id = $1
                AND ((a_type = 'spiral' AND a_id = $2)
                  OR (b_type = 'spiral' AND b_id = $2))""",
            user["user_id"], spiral_id,
        )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Cross-type pairing
# ---------------------------------------------------------------------------
#
# A "pair" is a user-asserted link between any two archive items —
# spiral ↔ spiral, spiral ↔ text-check, text-check ↔ compatibility, etc.
# The pair row stores (a_type, a_id, b_type, b_id) without enforcing an
# order. On read we OR both sides so each item shows every link.

VALID_PAIR_TYPES = {"spiral", "text_check", "compatibility"}


class PairCreate(BaseModel):
    a_type: str
    a_id: str
    b_type: str
    b_id: str
    note: Optional[str] = None


async def _verify_item_owned(conn, item_type: str, item_id: str, user_id: str) -> bool:
    """Confirm the requested item belongs to the user. Prevents pairing
    your own item with a stranger's id."""
    table = {
        "spiral": "spirals",
        "text_check": "text_checks",
        "compatibility": "compatibility_tests",
    }.get(item_type)
    if not table:
        return False
    row = await conn.fetchrow(
        f"SELECT 1 FROM {table} WHERE id = $1 AND user_id = $2",
        item_id, user_id,
    )
    return row is not None


@app.post("/api/pairs")
async def create_pair(body: PairCreate, user: dict = Depends(get_current_user)):
    db_required()
    await _ensure_archive_tables()
    if body.a_type not in VALID_PAIR_TYPES or body.b_type not in VALID_PAIR_TYPES:
        raise HTTPException(400, "Invalid item type")
    if body.a_type == body.b_type and body.a_id == body.b_id:
        raise HTTPException(400, "Cannot pair an item with itself")
    async with pool.acquire() as conn:
        # Ownership check on both sides.
        ok_a = await _verify_item_owned(conn, body.a_type, body.a_id, user["user_id"])
        ok_b = await _verify_item_owned(conn, body.b_type, body.b_id, user["user_id"])
        if not (ok_a and ok_b):
            raise HTTPException(404, "One or both items not found")
        # De-dup: if a pair (either orientation) already exists, return it.
        existing = await conn.fetchrow(
            """SELECT * FROM item_pairs
                WHERE user_id = $1
                  AND ((a_type = $2 AND a_id = $3 AND b_type = $4 AND b_id = $5)
                    OR (a_type = $4 AND a_id = $5 AND b_type = $2 AND b_id = $3))""",
            user["user_id"], body.a_type, body.a_id, body.b_type, body.b_id,
        )
        if existing:
            return {"pair": dict(existing), "created": False}
        pair_id = f"pr_{uuid.uuid4().hex[:14]}"
        await conn.execute(
            """INSERT INTO item_pairs (id, user_id, a_type, a_id, b_type, b_id, note, created_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())""",
            pair_id, user["user_id"],
            body.a_type, body.a_id, body.b_type, body.b_id,
            (body.note or "").strip()[:200] or None,
        )
        row = await conn.fetchrow("SELECT * FROM item_pairs WHERE id = $1", pair_id)
    return {"pair": dict(row), "created": True}


@app.delete("/api/pairs/{pair_id}")
async def delete_pair(pair_id: str, user: dict = Depends(get_current_user)):
    db_required()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM item_pairs WHERE id = $1 AND user_id = $2",
            pair_id, user["user_id"],
        )
    if result.endswith("0"):
        raise HTTPException(404, "Pair not found")
    return {"ok": True}


@app.get("/api/pairs/{item_type}/{item_id}")
async def list_pairs_for_item(item_type: str, item_id: str, user: dict = Depends(get_current_user)):
    """Return every pair row that references the given item, with the
    partner item denormalised into the payload so the frontend can render
    a one-glance list without N+1 follow-up requests."""
    db_required()
    await _ensure_archive_tables()
    if item_type not in VALID_PAIR_TYPES:
        raise HTTPException(400, "Invalid item type")
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM item_pairs
                WHERE user_id = $1
                  AND ((a_type = $2 AND a_id = $3)
                    OR (b_type = $2 AND b_id = $3))
                ORDER BY created_at DESC""",
            user["user_id"], item_type, item_id,
        )
        pairs = []
        for r in rows:
            d = dict(r)
            # Figure out which side is "the other one" from this item's POV.
            if d["a_type"] == item_type and d["a_id"] == item_id:
                partner_type, partner_id = d["b_type"], d["b_id"]
            else:
                partner_type, partner_id = d["a_type"], d["a_id"]
            d["partner_type"] = partner_type
            d["partner_id"]   = partner_id

            # Resolve the partner's display metadata (name + preview) so
            # the frontend doesn't have to fan out fetches.
            preview: Optional[Dict[str, Any]] = None
            if partner_type == "spiral":
                pr = await conn.fetchrow(
                    "SELECT id, name, situation_text, category, created_at FROM spirals WHERE id = $1 AND user_id = $2",
                    partner_id, user["user_id"],
                )
                if pr:
                    preview = {
                        "id": pr["id"],
                        "name": pr["name"],
                        "title": (pr["name"] or pr["situation_text"] or "")[:80],
                        "subtitle": pr["category"],
                        "created_at": pr["created_at"].isoformat() if isinstance(pr["created_at"], datetime) else pr["created_at"],
                    }
            elif partner_type == "text_check":
                pr = await conn.fetchrow(
                    "SELECT id, name, draft, relationship, created_at FROM text_checks WHERE id = $1 AND user_id = $2",
                    partner_id, user["user_id"],
                )
                if pr:
                    preview = {
                        "id": pr["id"],
                        "name": pr["name"],
                        "title": (pr["name"] or pr["draft"] or "")[:80],
                        "subtitle": f"Text → {pr['relationship'] or 'someone'}",
                        "created_at": pr["created_at"].isoformat() if isinstance(pr["created_at"], datetime) else pr["created_at"],
                    }
            elif partner_type == "compatibility":
                pr = await conn.fetchrow(
                    "SELECT id, name, person_a, person_b, created_at FROM compatibility_tests WHERE id = $1 AND user_id = $2",
                    partner_id, user["user_id"],
                )
                if pr:
                    pa = pr["person_a"] if isinstance(pr["person_a"], dict) else (json.loads(pr["person_a"]) if pr["person_a"] else {})
                    pb = pr["person_b"] if isinstance(pr["person_b"], dict) else (json.loads(pr["person_b"]) if pr["person_b"] else {})
                    preview = {
                        "id": pr["id"],
                        "name": pr["name"],
                        "title": pr["name"] or f"{pa.get('name','?')} × {pb.get('name','?')}",
                        "subtitle": "Compatibility",
                        "created_at": pr["created_at"].isoformat() if isinstance(pr["created_at"], datetime) else pr["created_at"],
                    }
            d["preview"] = preview
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = d["created_at"].isoformat()
            # Drop the orphan if the partner item no longer exists.
            if preview is not None:
                pairs.append(d)
    return {"pairs": pairs}


# ---------------------------------------------------------------------------
# Tasks / Level / Activity
# ---------------------------------------------------------------------------

@app.get("/api/tasks")
async def get_tasks(user: dict = Depends(get_current_user)):
    if user.get("is_guest") or (user.get("plan_tier") or "free") == "free":
        raise HTTPException(402, "Pro required for tasks")
    await ensure_task_rows(user["user_id"])
    day = current_daily_period()
    week = current_weekly_period()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM user_tasks WHERE user_id = $1 AND period IN ($2,$3)",
            user["user_id"], day, week,
        )
    by_id = {r["task_id"]: dict(r) for r in rows}
    def enrich(defs, period):
        out = []
        for d in defs:
            r = by_id.get(d["task_id"]) or {}
            if r.get("period") != period:
                r = {}
            out.append({
                "task_id": d["task_id"],
                "label": d["label"],
                "xp": d["xp"],
                "target": d["target"],
                "progress": r.get("progress", 0),
                "claimed": bool(r.get("claimed")),
                "period": period,
            })
        return out
    return {
        "daily": enrich(DAILY_TASKS, day),
        "weekly": enrich(WEEKLY_TASKS, week),
    }


@app.post("/api/tasks/{task_id}/claim")
async def claim_task(task_id: str, user: dict = Depends(get_current_user)):
    if user.get("is_guest") or (user.get("plan_tier") or "free") == "free":
        raise HTTPException(402, "Pro required for tasks")
    # Find the task definition for XP
    all_defs = {t["task_id"]: t for t in DAILY_TASKS + WEEKLY_TASKS}
    if task_id not in all_defs:
        raise HTTPException(404, "Unknown task")
    period = current_daily_period() if task_id in {t["task_id"] for t in DAILY_TASKS} else current_weekly_period()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM user_tasks WHERE user_id = $1 AND task_id = $2 AND period = $3",
            user["user_id"], task_id, period,
        )
        if not row:
            raise HTTPException(404, "Task not found")
        if row["claimed"]:
            raise HTTPException(400, "Already claimed")
        if (row["progress"] or 0) < row["target"]:
            raise HTTPException(400, "Task not complete")
        await conn.execute(
            "UPDATE user_tasks SET claimed = TRUE, claimed_at = $1 WHERE id = $2",
            now_iso(), row["id"],
        )
    granted = await grant_xp(user["user_id"], all_defs[task_id]["xp"])
    return {"claimed": True, "xp_granted": all_defs[task_id]["xp"], **granted}


# ---------------------------------------------------------------------------
# Thinkpass+ tier rewards. Every 5 levels = a claimable cosmetic. Unlike
# legacy unlocks (auto-granted), tiers must be explicitly claimed by the user
# from the Thinkpass page. Claiming adds the item to unlocked_items AND
# records the tier level in claimed_tiers so we can show "Owned" state.
# ---------------------------------------------------------------------------

@app.get("/api/thinkpass/tiers")
async def list_tiers(user: dict = Depends(get_current_user)):
    db_required()
    claimed = user.get("claimed_tiers") or []
    if isinstance(claimed, str):
        try: claimed = json.loads(claimed)
        except Exception: claimed = []
    current_level = user.get("level") or 1
    out = []
    for lv, kind, value, label in LEVEL_UNLOCKS:
        out.append({
            "level": lv,
            "kind": kind,
            "value": value,
            "label": label,
            "claimed": lv in claimed,
            "ready": current_level >= lv and lv not in claimed,
            "locked": current_level < lv,
        })
    return {"tiers": out, "current_level": current_level}


@app.post("/api/thinkpass/claim/{tier_level}")
async def claim_tier(tier_level: int, user: dict = Depends(get_current_user)):
    db_required()
    match = next(((lv, k, v, lbl) for (lv, k, v, lbl) in LEVEL_UNLOCKS if lv == tier_level), None)
    if not match:
        raise HTTPException(404, "No tier at that level")
    lv, kind, value, _label = match
    if (user.get("level") or 1) < lv:
        raise HTTPException(400, f"Level {lv} required")

    claimed_tiers = user.get("claimed_tiers") or []
    if isinstance(claimed_tiers, str):
        try: claimed_tiers = json.loads(claimed_tiers)
        except Exception: claimed_tiers = []
    if lv in claimed_tiers:
        raise HTTPException(400, "Already claimed")
    claimed_tiers.append(lv)

    unlocked = user.get("unlocked_items") or []
    if isinstance(unlocked, str):
        try: unlocked = json.loads(unlocked)
        except Exception: unlocked = []
    item_id = f"{kind}:{value}"
    if item_id not in unlocked:
        unlocked.append(item_id)

    async with pool.acquire() as conn:
        await conn.execute(
            """UPDATE users
                  SET claimed_tiers = $1, unlocked_items = $2
                WHERE user_id = $3""",
            json.dumps(claimed_tiers), json.dumps(unlocked), user["user_id"],
        )
    return {"ok": True, "claimed_tier": lv, "item": item_id, "unlocked_items": unlocked}


@app.get("/api/level")
async def get_level(user: dict = Depends(get_current_user)):
    xp = user.get("xp") or 0
    level = level_from_xp(xp)
    start_xp = total_xp_for_level(level)
    end_xp = total_xp_for_level(level + 1) if level < 100 else start_xp
    xp_in_level = xp - start_xp
    xp_needed = max(end_xp - start_xp, 1)
    pct = round(min(xp_in_level / xp_needed, 1) * 100)

    unlocked = user.get("unlocked_items") or []
    if isinstance(unlocked, str):
        try:
            unlocked = json.loads(unlocked)
        except Exception:
            unlocked = []

    upcoming = []
    for lv, kind, value, label in LEVEL_UNLOCKS:
        if lv > level:
            upcoming.append({"level": lv, "kind": kind, "value": value, "label": label})
        if len(upcoming) >= 5:
            break

    return {
        "level": level,
        "xp": xp,
        "xp_in_level": xp_in_level,
        "xp_needed": xp_needed,
        "pct": pct,
        "unlocked_items": unlocked,
        "upcoming_unlocks": upcoming,
        "all_unlocks": [
            {"level": lv, "kind": k, "value": v, "label": label}
            for lv, k, v, label in LEVEL_UNLOCKS
        ],
    }


@app.get("/api/activity")
async def get_activity(
    user: dict = Depends(get_current_user),
    range_: str = "week",
):
    today = datetime.now(timezone.utc).date()
    if range_ == "year":
        start = today - timedelta(days=365)
        bucket = "month"
    elif range_ == "month":
        start = today - timedelta(days=30)
        bucket = "day"
    else:
        start = today - timedelta(days=7)
        bucket = "day"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT created_at FROM spirals WHERE user_id = $1 AND created_at >= $2",
            user["user_id"], start.isoformat(),
        )
    buckets: dict[str, int] = {}
    for r in rows:
        try:
            d = datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00")).date()
        except Exception:
            continue
        key = d.isoformat() if bucket == "day" else f"{d.year}-{d.month:02d}"
        buckets[key] = buckets.get(key, 0) + 1
    return {"range": range_, "buckets": [{"date": k, "count": v} for k, v in sorted(buckets.items())]}


@app.get("/api/profile/stats")
async def profile_stats(user: dict = Depends(get_current_user)):
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM spirals WHERE user_id = $1", user["user_id"],
        )
        resolved = await conn.fetchval(
            "SELECT COUNT(*) FROM spirals WHERE user_id = $1 AND resolved = TRUE",
            user["user_id"],
        )
        by_tone = await conn.fetch(
            "SELECT tone_used, COUNT(*) c FROM spirals WHERE user_id = $1 GROUP BY tone_used",
            user["user_id"],
        )
        by_cat = await conn.fetch(
            "SELECT category, COUNT(*) c FROM spirals WHERE user_id = $1 GROUP BY category",
            user["user_id"],
        )
    total = total or 0
    resolved = resolved or 0
    return {
        "total": total,
        "resolved": resolved,
        "resolved_pct": round((resolved / total) * 100) if total else 0,
        "streak": user.get("streak_count") or 0,
        "xp": user.get("xp") or 0,
        "level": user.get("level") or 1,
        "by_tone": {r["tone_used"]: r["c"] for r in by_tone},
        "by_category": {r["category"]: r["c"] for r in by_cat},
        "shared_count": user.get("shared_count") or 0,
    }


@app.get("/api/profile/full_stats")
async def profile_full_stats(user: dict = Depends(get_current_user)):
    """Everything for the Stats screen — every counter we can compute from
    the user's spirals plus the user record itself."""
    db_required()
    uid = user["user_id"]
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM spirals WHERE user_id = $1", uid,
        ) or 0
        resolved = await conn.fetchval(
            """SELECT COUNT(*) FROM spirals
               WHERE user_id = $1 AND resolved = TRUE
                 AND (resolution_status IS NULL
                      OR resolution_status NOT IN ('plot_twist_good','plot_twist_bad'))""",
            uid,
        ) or 0
        not_resolved = await conn.fetchval(
            """SELECT COUNT(*) FROM spirals
               WHERE user_id = $1 AND resolved = FALSE
                 AND resolution_status IS NOT NULL
                 AND resolution_status NOT IN ('plot_twist_good','plot_twist_bad')""",
            uid,
        ) or 0
        plot_twist_good = await conn.fetchval(
            "SELECT COUNT(*) FROM spirals WHERE user_id = $1 AND resolution_status = 'plot_twist_good'",
            uid,
        ) or 0
        plot_twist_bad = await conn.fetchval(
            "SELECT COUNT(*) FROM spirals WHERE user_id = $1 AND resolution_status = 'plot_twist_bad'",
            uid,
        ) or 0
        flagged = await conn.fetchval(
            "SELECT COUNT(*) FROM spirals WHERE user_id = $1 AND flagged = TRUE", uid,
        ) or 0
        in_folders = await conn.fetchval(
            "SELECT COUNT(*) FROM spirals WHERE user_id = $1 AND folder_id IS NOT NULL", uid,
        ) or 0
        folder_count = await conn.fetchval(
            "SELECT COUNT(*) FROM folders WHERE user_id = $1", uid,
        ) or 0

        # Total words across all situation_text
        rows = await conn.fetch(
            "SELECT situation_text, created_at FROM spirals WHERE user_id = $1", uid,
        )
        total_words = sum(
            len((r["situation_text"] or "").split()) for r in rows
        )

        # Peak day — group spirals by date, find the day with the most
        peak_count = 0
        peak_day: Optional[str] = None
        day_counts: dict[str, int] = {}
        for r in rows:
            try:
                d = datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00")).date().isoformat()
                day_counts[d] = day_counts.get(d, 0) + 1
                if day_counts[d] > peak_count:
                    peak_count = day_counts[d]
                    peak_day = d
            except Exception:
                continue

        by_tone = await conn.fetch(
            "SELECT tone_used, COUNT(*) c FROM spirals WHERE user_id = $1 GROUP BY tone_used", uid,
        )
        by_cat = await conn.fetch(
            "SELECT category, COUNT(*) c FROM spirals WHERE user_id = $1 GROUP BY category", uid,
        )

        active_days = await conn.fetchval(
            "SELECT COUNT(DISTINCT DATE(created_at)) FROM spirals WHERE user_id = $1", uid,
        ) or 0

    return {
        "spirals_created": total,
        "spirals_deleted": user.get("deleted_count") or 0,
        "spirals_saved": total,  # currently every spiral that exists is saved
        "resolved": resolved,
        "not_resolved": not_resolved,
        "plot_twist_good": plot_twist_good,
        "plot_twist_bad": plot_twist_bad,
        "plot_twist_total": plot_twist_good + plot_twist_bad,
        "flagged": flagged,
        "in_folders": in_folders,
        "folder_count": folder_count,
        "total_words": total_words,
        "peak_day": peak_day,
        "peak_day_count": peak_count,
        "active_days": active_days,
        "shared_count": user.get("shared_count") or 0,
        "current_streak": user.get("streak_count") or 0,
        "xp": user.get("xp") or 0,
        "level": user.get("level") or 1,
        "by_tone": {r["tone_used"]: r["c"] for r in by_tone},
        "by_category": {r["category"]: r["c"] for r in by_cat},
    }


@app.get("/api/wrapped/current")
async def wrapped_current(user: dict = Depends(get_current_user)):
    if user.get("is_guest") or (user.get("plan_tier") or "free") == "free":
        raise HTTPException(402, "Pro required for wrapped")
    today = datetime.now(timezone.utc).date()
    start = today.replace(day=1)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM spirals WHERE user_id = $1 AND created_at >= $2",
            user["user_id"], start.isoformat(),
        )
    spirals = [dict(r) for r in rows]
    top_cat = {}
    top_tone = {}
    resolved_count = 0
    for s in spirals:
        top_cat[s["category"]] = top_cat.get(s["category"], 0) + 1
        top_tone[s["tone_used"]] = top_tone.get(s["tone_used"], 0) + 1
        if s.get("resolved"):
            resolved_count += 1
    return {
        "period_start": start.isoformat(),
        "spirals": len(spirals),
        "resolved": resolved_count,
        "top_category": max(top_cat.items(), key=lambda x: x[1])[0] if top_cat else None,
        "top_tone": max(top_tone.items(), key=lambda x: x[1])[0] if top_tone else None,
    }


# ---------------------------------------------------------------------------
# Insights — "Your Recurring Loops"
#
# Pattern detection across the user's archive. This is the differentiator
# vs. a generic chatbot: ChatGPT forgets you between sessions; this
# endpoint surfaces "you've spiralled about Mira 7 times this month — 6
# of those resolved as nothing happened."
#
# Clustering algorithm (deliberately simple v1):
#   • Group by category, then sub-group by tag overlap (Jaccard ≥ 0.34).
#   • A "loop" needs ≥ 3 spirals in the same cluster.
#   • Per loop we compute: count, last_seen, top tags, resolution
#     breakdown, and an "accuracy %" (= fraction that resolved as
#     resolved/plot_twist_good vs not_resolved/plot_twist_bad — a proxy
#     for "your worst-case prediction came true").
#
# Free users get one preview loop with the rest locked. Pro users get
# everything. We always run the full computation server-side and trim
# the response — that way Pro upgrades surface results immediately
# without re-querying.
# ---------------------------------------------------------------------------

def _tag_key(tags) -> tuple:
    """Normalise a spiral's tags into a sorted tuple for hashing."""
    if not tags:
        return ()
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            return ()
    return tuple(sorted(t.strip().lower() for t in tags if t and isinstance(t, str)))


def _jaccard(a: tuple, b: tuple) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cluster_spirals(spirals: list) -> list:
    """Greedy single-pass clustering: each spiral joins the first existing
    cluster whose representative shares its category AND ≥ 0.34 Jaccard
    overlap on tags. Otherwise it seeds a new cluster.

    Returns a list of clusters; each cluster is a list of spiral dicts.
    Deliberately simple — no embeddings, no transitive merging. Good
    enough to surface "you keep spiralling about X" without an LLM bill.
    """
    clusters: list = []
    for s in spirals:
        cat = s.get("category") or "other"
        tags = _tag_key(s.get("tags"))
        joined = False
        for cl in clusters:
            rep = cl[0]
            if (rep.get("category") or "other") != cat:
                continue
            rep_tags = _tag_key(rep.get("tags"))
            # Empty-tag spirals only cluster with other empty-tag spirals
            # in the same category — otherwise everything with no tags
            # lumps together and obscures the real pattern.
            if not tags and not rep_tags:
                cl.append(s)
                joined = True
                break
            if _jaccard(tags, rep_tags) >= 0.34:
                cl.append(s)
                joined = True
                break
        if not joined:
            clusters.append([s])
    return clusters


def _summarise_loop(cluster: list) -> dict:
    """Compute display stats for one cluster of spirals."""
    from collections import Counter
    cat = cluster[0].get("category") or "other"
    tag_counts: Counter = Counter()
    res_counts: Counter = Counter()
    last_seen = None
    sample_situations: list = []
    for s in cluster:
        tags = _tag_key(s.get("tags"))
        for t in tags:
            tag_counts[t] += 1
        rs = s.get("resolution_status") or ("resolved" if s.get("resolved") else None)
        if rs:
            res_counts[rs] += 1
        created = s.get("created_at")
        if created and (last_seen is None or created > last_seen):
            last_seen = created
        # Keep first 80 chars of up to 3 sample situations for the
        # "what these have in common" preview text.
        if len(sample_situations) < 3:
            txt = (s.get("situation_text") or "").strip().replace("\n", " ")
            if txt:
                sample_situations.append(txt[:80] + ("…" if len(txt) > 80 else ""))
    top_tags = [t for t, _ in tag_counts.most_common(3)]
    total_resolved = sum(res_counts.values())
    # "Brain accuracy" = how often you were RIGHT to worry.
    # bad outcomes (not_resolved / plot_twist_bad) / total_resolved
    bad = res_counts.get("not_resolved", 0) + res_counts.get("plot_twist_bad", 0)
    accuracy_pct = None
    if total_resolved >= 2:
        accuracy_pct = round((bad / total_resolved) * 100)
    # Generate a short headline: top tag if any, else category.
    headline = top_tags[0].title() if top_tags else cat.title()
    return {
        "headline": headline,
        "category": cat,
        "count": len(cluster),
        "tags": top_tags,
        "last_seen": last_seen,
        "resolution_breakdown": dict(res_counts),
        "resolved_count": total_resolved,
        "worry_accuracy_pct": accuracy_pct,
        "sample_situations": sample_situations,
    }


@app.get("/api/insights/checkin_candidate")
async def insights_checkin_candidate(user: dict = Depends(get_current_user)):
    """One unresolved spiral worth checking in on — picked by the local
    notification scheduler to personalise the daily nudge.

    Heuristic: oldest unresolved spiral that's at least 24h old (long
    enough for real life to have happened) and at most 30 days old (any
    older and the user has probably moved on; nagging would feel weird).
    Returns {"spiral": null} when nothing qualifies — the frontend then
    falls back to the generic "did you spiral today?" reminder.
    """
    db_required()
    is_pro_user = (user.get("plan_tier") or "free") in {"pro_weekly", "pro_monthly", "pro_yearly", "lifetime"}
    if not is_pro_user:
        # Free users can still call the endpoint without breaking — they
        # just always get null so the frontend keeps using the generic
        # reminder. Avoids a 403 round-trip on every app open.
        return {"spiral": None, "is_pro": False}
    now = datetime.now(timezone.utc)
    cutoff_recent = (now - timedelta(hours=24)).isoformat()
    cutoff_stale = (now - timedelta(days=30)).isoformat()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, name, situation_text, category, created_at
               FROM spirals
               WHERE user_id = $1
                 AND status = 'complete'
                 AND (resolved IS FALSE OR resolved IS NULL)
                 AND (resolution_status IS NULL
                      OR resolution_status NOT IN
                         ('resolved','not_resolved','other',
                          'plot_twist_good','plot_twist_bad'))
                 AND created_at < $2
                 AND created_at > $3
               ORDER BY created_at ASC
               LIMIT 1""",
            user["user_id"], cutoff_recent, cutoff_stale,
        )
    if not row:
        return {"spiral": None, "is_pro": True}
    s = dict(row)
    # Headline: prefer the user's chosen name; fall back to a short
    # snippet of the situation text. Used verbatim in the notification.
    raw_name = s.get("name")
    if raw_name and raw_name.strip():
        headline = raw_name.strip()[:48]
    else:
        snippet = (s.get("situation_text") or "").strip().replace("\n", " ")
        headline = (snippet[:48] + "…") if len(snippet) > 48 else (snippet or "that thing")
    return {
        "spiral": {
            "id": s["id"],
            "name": headline,
            "category": s["category"],
            "created_at": s["created_at"].isoformat() if hasattr(s["created_at"], "isoformat") else s["created_at"],
        },
        "is_pro": True,
    }


@app.get("/api/insights/score")
async def insights_score(user: dict = Depends(get_current_user)):
    """Loop Resolution Score — your personal "your brain was wrong N%
    of the time" report card. Aggregated across every resolved spiral
    the user has.

    Response shape:
      {
        "total_spirals": int,
        "resolved_count": int,
        # The headline stat. Higher = your worst-case predictions
        # mostly didn't come true (good news, brain is wrong a lot).
        "worst_case_avoided_pct": int | null,
        # Granular resolution breakdown (one of: resolved / not_resolved /
        # plot_twist_good / plot_twist_bad / other).
        "resolution_breakdown": { ... },
        # Pro-only: where your brain was most wrong vs. most right.
        "best_tone": str | null,
        "worst_category": str | null,   # category with highest miss rate
        "best_category": str | null,    # category with most correct predictions
        "plot_twist_count": int,
        "share_line": str,              # one-liner ready for the share card
        "is_pro": bool,
        "preview_only": bool,
      }
    """
    db_required()
    is_pro_user = (user.get("plan_tier") or "free") in {"pro_weekly", "pro_monthly", "pro_yearly", "lifetime"}
    # Hard-lock for free users — skip the aggregation entirely. We still
    # return total_spirals so the upsell hero can show "you have N
    # receipts to tally — Pro unlocks the score". Everything else is
    # null and the frontend renders the locked hero.
    if not is_pro_user:
        async with pool.acquire() as conn:
            count_row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM spirals WHERE user_id = $1 AND status = 'complete'",
                user["user_id"],
            )
        return {
            "total_spirals": (count_row["n"] if count_row else 0),
            "resolved_count": 0,
            "worst_case_avoided_pct": None,
            "resolution_breakdown": {},
            "best_tone": None,
            "worst_category": None,
            "best_category": None,
            "plot_twist_count": 0,
            "share_line": "Unlock the receipts with Pro.",
            "is_pro": False,
            "preview_only": True,
        }
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT category, tone_used, resolved, resolution_status
               FROM spirals
               WHERE user_id = $1 AND status = 'complete'""",
            user["user_id"],
        )
    spirals = [dict(r) for r in rows]
    total = len(spirals)

    # A spiral counts as "resolved" if it has any resolution_status
    # (resolved / not_resolved / plot_twist_good / plot_twist_bad / other).
    # The resolved boolean column is older + redundant, so we treat any
    # resolution_status as truth.
    from collections import Counter
    res_counter: Counter = Counter()
    # Category-level miss tracking: per-category counts of correct vs.
    # wrong predictions. "Correct prediction" = your worst-case did NOT
    # come true → resolved as resolved / plot_twist_good.
    cat_right: Counter = Counter()
    cat_wrong: Counter = Counter()
    tone_right: Counter = Counter()
    tone_total: Counter = Counter()
    plot_twist_count = 0

    for s in spirals:
        rs = s.get("resolution_status")
        if not rs:
            continue
        res_counter[rs] += 1
        is_right = rs in {"resolved", "plot_twist_good"}
        is_wrong = rs in {"not_resolved", "plot_twist_bad"}
        if rs in {"plot_twist_good", "plot_twist_bad"}:
            plot_twist_count += 1
        cat = s.get("category") or "other"
        tone = s.get("tone_used") or "balanced"
        if is_right:
            cat_right[cat] += 1
        if is_wrong:
            cat_wrong[cat] += 1
        if is_right:
            tone_right[tone] += 1
        if is_right or is_wrong:
            tone_total[tone] += 1

    resolved_count = sum(res_counter.values())
    right_count = sum(cat_right.values())
    decisive_count = right_count + sum(cat_wrong.values())
    worst_case_avoided_pct = (
        round((right_count / decisive_count) * 100) if decisive_count > 0 else None
    )

    # Per-category miss rate (lower = your worry rate of return is good).
    # We only consider categories with ≥ 2 resolutions to avoid one-shot
    # noise pretending to be a pattern.
    def _cat_miss_rate(cat: str) -> float:
        r, w = cat_right[cat], cat_wrong[cat]
        total_cat = r + w
        return (w / total_cat) if total_cat >= 2 else -1.0  # -1 → ignore

    cats_decisive = [c for c in set(list(cat_right) + list(cat_wrong)) if _cat_miss_rate(c) >= 0]
    worst_category = max(cats_decisive, key=_cat_miss_rate) if cats_decisive else None
    best_category = min(cats_decisive, key=_cat_miss_rate) if cats_decisive else None

    # Most-trusted tone: highest "right rate" with ≥ 2 resolutions.
    def _tone_right_rate(t: str) -> float:
        total_t = tone_total[t]
        return (tone_right[t] / total_t) if total_t >= 2 else -1.0
    tones_decisive = [t for t in tone_total if _tone_right_rate(t) >= 0]
    best_tone = max(tones_decisive, key=_tone_right_rate) if tones_decisive else None

    # Build a punchy one-line headline for the share card. Falls back
    # gracefully when there aren't enough resolutions to compute it.
    if worst_case_avoided_pct is None:
        share_line = "Still tracking — resolve more spirals to see your number."
    elif worst_case_avoided_pct >= 70:
        share_line = f"My brain was wrong {worst_case_avoided_pct}% of the time."
    elif worst_case_avoided_pct >= 50:
        share_line = f"My worst case missed more than it landed ({worst_case_avoided_pct}%)."
    else:
        share_line = f"My brain was actually right {100 - worst_case_avoided_pct}% of the time. Rude."

    return {
        "total_spirals": total,
        "resolved_count": resolved_count,
        "worst_case_avoided_pct": worst_case_avoided_pct,
        "resolution_breakdown": dict(res_counter),
        "best_tone": best_tone,
        "worst_category": worst_category,
        "best_category": best_category,
        "plot_twist_count": plot_twist_count,
        "share_line": share_line,
        "is_pro": is_pro_user,
        # Free users see the headline %, total count, and the share
        # line. Per-category / per-tone breakdown stays behind preview_only.
        "preview_only": not is_pro_user,
    }


@app.get("/api/insights/loops")
async def insights_loops(user: dict = Depends(get_current_user)):
    """Surface the user's recurring overthinking patterns.

    Response shape:
      {
        "loops": [ { headline, category, count, tags, last_seen,
                     resolution_breakdown, resolved_count,
                     worry_accuracy_pct, sample_situations }, ... ],
        "min_spirals_needed": 3,
        "total_spirals": int,
        "is_pro": bool,
        "preview_only": bool   (true when user is free — only first loop
                                is fully revealed in the UI),
      }
    """
    db_required()
    is_pro_user = (user.get("plan_tier") or "free") in {"pro_weekly", "pro_monthly", "pro_yearly", "lifetime"}
    # Hard-lock for free users — we don't even run the clustering.
    # Frontend uses is_pro=false to render the Pro upsell hero.
    # total_spirals is still returned so the upsell can say something
    # honest like "you have 47 spirals — Pro turns them into patterns".
    if not is_pro_user:
        async with pool.acquire() as conn:
            count_row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM spirals WHERE user_id = $1 AND status = 'complete'",
                user["user_id"],
            )
        return {
            "loops": [],
            "min_spirals_needed": 3,
            "total_spirals": (count_row["n"] if count_row else 0),
            "is_pro": False,
            "preview_only": True,
        }
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, category, tags, situation_text, resolved,
                      resolution_status, created_at
               FROM spirals
               WHERE user_id = $1 AND status = 'complete'
               ORDER BY created_at DESC""",
            user["user_id"],
        )
    spirals = [dict(r) for r in rows]
    clusters = [c for c in _cluster_spirals(spirals) if len(c) >= 3]
    clusters.sort(key=lambda cl: (-len(cl), -(len(cl[0].get("created_at") or ""))))
    loops = [_summarise_loop(cl) for cl in clusters[:8]]
    return {
        "loops": loops,
        "min_spirals_needed": 3,
        "total_spirals": len(spirals),
        "is_pro": True,
        "preview_only": False,
    }


# ---------------------------------------------------------------------------
# Payments — Stripe Checkout
# ---------------------------------------------------------------------------

PACKAGES = {
    "weekly":  {"id": "weekly",  "label": "Spiral Starter",      "amount": 1.99,  "currency": "usd", "tier": "pro_weekly"},
    "monthly": {"id": "monthly", "label": "Deep In My Feelings", "amount": 5.99,  "currency": "usd", "tier": "pro_monthly"},
    "lifetime":{"id": "lifetime","label": "Infinite Loop Pass",  "amount": 29.99, "currency": "usd", "tier": "lifetime"},
}
# NOTE: pro_yearly tier handling is kept downstream (apply_plan,
# isPro checks) for back-compat in case any user briefly bought it
# before this revert. New checkouts can't reach it.


# ---------------------------------------------------------------------------
# AI diagnostic — hit this to verify Gemini is actually responding.
#   GET /api/diag/ai     → quick connectivity test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DEV ONLY — testing shortcut so we can exercise Pro features without doing
# the full Stripe checkout dance. Set the caller's plan_tier to "lifetime"
# and grant a chunk of XP so customisation unlocks become visible.
#
# Remove or gate behind a stronger check before shipping.
# ---------------------------------------------------------------------------

@app.post("/api/dev/grant_pro")
async def dev_grant_pro(user: dict = Depends(get_current_user)):
    """Only flips plan_tier to lifetime (so Pro gating passes) and clears
    is_guest. XP/level/cosmetics stay UNTOUCHED — so the user still has the
    real progression-from-zero experience for testing tasks and Thinkpass+."""
    db_required()
    async with pool.acquire() as conn:
        await conn.execute(
            """UPDATE users
                  SET plan_tier = 'lifetime',
                      plan_expires_at = NULL,
                      is_guest = FALSE
                WHERE user_id = $1""",
            user["user_id"],
        )
        row = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user["user_id"])
    return {"user": user_public(dict(row))}


@app.post("/api/dev/reset_xp")
async def dev_reset_xp(user: dict = Depends(get_current_user)):
    """Reset XP/level/claimed-tiers/unlocked-cosmetics to zero so the user
    can farm leveling from scratch. Plan tier stays."""
    db_required()
    async with pool.acquire() as conn:
        await conn.execute(
            """UPDATE users
                  SET xp = 0, level = 1,
                      unlocked_items = '[]', claimed_tiers = '[]',
                      customization = NULL
                WHERE user_id = $1""",
            user["user_id"],
        )
        row = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user["user_id"])
    return {"user": user_public(dict(row))}


@app.post("/api/dev/revoke_pro")
async def dev_revoke_pro(user: dict = Depends(get_current_user)):
    db_required()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET plan_tier = 'free', plan_expires_at = NULL WHERE user_id = $1",
            user["user_id"],
        )
        row = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user["user_id"])
    return {"user": user_public(dict(row))}


@app.get("/api/diag/db")
async def diag_db():
    """Browser-accessible: confirm archive tables exist + report row counts.
    Lets us check from a phone whether text-check / compatibility saves
    are actually landing in the DB."""
    if pool is None:
        return {"ok": False, "reason": "no_pool", "hint": "Set DATABASE_URL in backend/.env"}
    # Try the lazy ensure first — if lifespan never ran, this creates
    # the tables on the fly.
    try:
        await _ensure_archive_tables()
    except Exception as exc:
        return {"ok": False, "reason": "ensure_failed", "error": f"{type(exc).__name__}: {exc}"}
    out: Dict[str, Any] = {"ok": True, "tables": {}}
    async with pool.acquire() as conn:
        for tbl in ("spirals", "folders", "text_checks", "compatibility_tests", "item_pairs", "users"):
            try:
                row = await conn.fetchrow(f"SELECT COUNT(*)::int AS n FROM {tbl}")
                out["tables"][tbl] = {"exists": True, "rows": int(row["n"]) if row else 0}
            except Exception as exc:
                out["tables"][tbl] = {"exists": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
                out["ok"] = False
    return out


@app.get("/api/diag/ai")
async def diag_ai():
    """Browser-accessible diagnostic. Iterates through the model fallback
    chain and reports which model(s) responded. Returns a per-model status."""
    if not GEMINI_API_KEY:
        return {"ok": False, "reason": "no_api_key", "hint": "Set GEMINI_API_KEY in backend/.env"}
    try:
        from google import genai
        from google.genai import types as gen_types
        client = genai.Client(api_key=GEMINI_API_KEY)
        config = gen_types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=256,
        )
        prompt = (
            'Respond ONLY with valid JSON of the form '
            '{"ping":"pong","seen":"ALPHA-7"} and nothing else.'
        )
        per_model = []
        first_ok = None
        for model_name in GEMINI_MODELS:
            try:
                resp = await client.aio.models.generate_content(
                    model=model_name, contents=prompt, config=config,
                )
                raw = (resp.text or "").strip()
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None
                per_model.append({"model": model_name, "ok": True, "raw": raw[:200], "parsed": parsed})
                if first_ok is None and raw:
                    first_ok = model_name
            except Exception as me:
                per_model.append({
                    "model": model_name,
                    "ok": False,
                    "error_type": type(me).__name__,
                    "error_message": str(me)[:300],
                })
        return {
            "ok": first_ok is not None,
            "first_working_model": first_ok,
            "key_prefix": GEMINI_API_KEY[:8] + "…" if GEMINI_API_KEY else None,
            "per_model": per_model,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:400],
            "key_prefix": GEMINI_API_KEY[:8] + "…" if GEMINI_API_KEY else None,
        }


@app.get("/api/payments/packages")
async def list_packages():
    return {"packages": list(PACKAGES.values())}


def stripe_lib():
    """Lazy-load the stripe SDK. If the package isn't installed (or some
    transient import error happens), surface a 503 with a clear message so
    the caller sees 'Stripe SDK missing' instead of a generic 500."""
    try:
        import stripe as _stripe  # local import so the module is optional in dev
    except Exception as exc:
        print(f"[stripe_lib] import failed: {type(exc).__name__}: {exc}")
        raise HTTPException(503, f"Stripe SDK unavailable: {exc}")
    _stripe.api_key = STRIPE_API_KEY
    return _stripe


def _stripe_to_dict(obj) -> dict:
    """Coerce a stripe-python response (StripeObject, dict, or None) into a
    plain dict so call sites can use .get() / [key] without worrying about
    SDK version differences.

    Newer stripe-python (≥ 7.x) made StripeObject stop inheriting from
    dict, so direct .get() now triggers __getattr__ → AttributeError. The
    SDK exposes to_dict_recursive() for this exact case; older versions
    expose to_dict(). Fall back to the object's __dict__ as a last resort.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for meth in ("to_dict_recursive", "to_dict"):
        fn = getattr(obj, meth, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    # Last resort — most StripeObjects keep their fields in _previous or
    # the bare __dict__. This won't deep-resolve nested StripeObjects but
    # at least gives us top-level keys.
    try:
        return dict(obj)  # works if it does subclass dict on this SDK ver
    except Exception:
        return {k: v for k, v in getattr(obj, "__dict__", {}).items()
                if not k.startswith("_")}


@app.post("/api/payments/checkout")
async def create_checkout(body: CheckoutRequest, user: dict = Depends(get_current_user)):
    # Top-level try/except so NO unhandled exception ever escapes as a
    # generic 500. Every failure path is logged with a [create_checkout]
    # prefix and returned as a 502 with the real reason in the body.
    try:
        return await _create_checkout_inner(body, user)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[create_checkout] UNHANDLED {type(exc).__name__}: {exc}\n{tb}")
        raise HTTPException(502, f"Checkout failed ({type(exc).__name__}): {exc}")


async def _create_checkout_inner(body: CheckoutRequest, user: dict):
    import asyncio, time
    t0 = time.monotonic()
    print(f"[create_checkout] start user={user.get('user_id')} guest={user.get('is_guest')} pkg={body.package_id}")
    if user.get("is_guest"):
        raise HTTPException(403, "Sign in before upgrading")
    pkg = PACKAGES.get(body.package_id)
    if not pkg:
        raise HTTPException(400, "Unknown package")
    if not STRIPE_API_KEY:
        raise HTTPException(503, "Stripe not configured (STRIPE_API_KEY missing)")
    stripe = stripe_lib()
    print(f"[create_checkout] +{time.monotonic()-t0:.2f}s stripe_lib ready")
    # Stripe rejects custom-scheme URLs. Route through our own HTTPS bridge
    # that immediately redirects to the app's deep link.
    base_https = "https://overthink-this-api.onrender.com"
    success_url = f"{base_https}/api/payments/return?session_id={{CHECKOUT_SESSION_ID}}&action=success"
    cancel_url  = f"{base_https}/api/payments/return?action=cancel"
    # Weekly + monthly → recurring subscriptions. Lifetime → one-time.
    is_subscription = pkg["id"] in {"weekly", "monthly", "yearly"}
    interval = (
        "week"  if pkg["id"] == "weekly"  else
        "month" if pkg["id"] == "monthly" else
        "year"  if pkg["id"] == "yearly"  else
        None
    )
    price_data: dict = {
        "currency": pkg["currency"],
        "product_data": {"name": pkg["label"]},
        "unit_amount": int(round(pkg["amount"] * 100)),
    }
    if interval:
        price_data["recurring"] = {"interval": interval}

    # Build kwargs dict so we only pass `subscription_data` when it's actually
    # a subscription. Passing `subscription_data=None` to stripe-python on a
    # one-time payment makes Stripe reject the call with
    # "Received unknown parameter: subscription_data" — that was a real 500
    # source for the lifetime package.
    create_kwargs: dict = {
        "mode": "subscription" if is_subscription else "payment",
        "payment_method_types": ["card"],
        "line_items": [{"price_data": price_data, "quantity": 1}],
        "success_url": success_url,
        "cancel_url": cancel_url,
        "metadata": {"user_id": user["user_id"], "package_id": pkg["id"]},
    }
    if is_subscription:
        # For subscriptions, attach user_id to the subscription too so the
        # webhook can find them on renewal events.
        create_kwargs["subscription_data"] = {
            "metadata": {"user_id": user["user_id"], "package_id": pkg["id"]},
        }

    try:
        # stripe-python is SYNCHRONOUS — calling it directly inside an
        # async endpoint blocks the asyncio event loop. On a 0.1 CPU
        # Render instance with cold-start latency, that can push the
        # whole request past Render's ~100s timeout and the client sees
        # a 500 with no body. asyncio.to_thread moves the blocking call
        # to the default thread pool so the loop stays responsive.
        # Also wrap in a 25s timeout — if Stripe is slow, fail loud
        # instead of letting Render kill the whole request.
        print(f"[create_checkout] +{time.monotonic()-t0:.2f}s calling Stripe…")
        session = await asyncio.wait_for(
            asyncio.to_thread(stripe.checkout.Session.create, **create_kwargs),
            timeout=25.0,
        )
        print(f"[create_checkout] +{time.monotonic()-t0:.2f}s Stripe returned session={session.id}")
    except asyncio.TimeoutError:
        print(f"[create_checkout] Stripe call timed out after 25s (event loop / network)")
        raise HTTPException(504, "Stripe took too long to respond. Try again in a few seconds.")
    except Exception as exc:
        # Surface the real Stripe error message + type so the app shows a
        # specific reason instead of an opaque 500. stripe-python's errors
        # have .user_message / .code / .param attributes that are way more
        # useful than str(exc) alone.
        detail = (
            getattr(exc, "user_message", None)
            or getattr(exc, "_message", None)
            or str(exc)
            or type(exc).__name__
        )
        code = getattr(exc, "code", None) or getattr(exc, "type", None)
        print(f"[create_checkout] Stripe rejected request: type={type(exc).__name__} code={code} detail={detail}")
        raise HTTPException(502, f"Stripe error: {detail}")

    # The Stripe session is now created. Record it in our DB, but DON'T let
    # a DB hiccup block returning the checkout URL — the user has a valid
    # Stripe session and can still pay; the webhook / polling endpoint will
    # heal the missing row via apply_plan's metadata-reconstruction path.
    # 5s timeout on the DB write so a stuck pool can't take down checkout.
    try:
        if pool is None:
            print(f"[create_checkout] pool is None — skipping tx insert for session {session.id}")
        else:
            async def _ins():
                async with pool.acquire() as conn:
                    await conn.execute(
                        """INSERT INTO payment_transactions
                             (session_id, user_id, package_id, amount, currency, metadata, payment_status, status, created_at)
                           VALUES ($1,$2,$3,$4,$5,$6,'initiated','open',$7)
                           ON CONFLICT (session_id) DO NOTHING""",
                        session.id, user["user_id"], pkg["id"], pkg["amount"], pkg["currency"],
                        json.dumps({"package": pkg}), now_iso(),
                    )
            await asyncio.wait_for(_ins(), timeout=5.0)
            print(f"[create_checkout] +{time.monotonic()-t0:.2f}s tx row inserted")
    except asyncio.TimeoutError:
        print(f"[create_checkout] DB insert TIMED OUT for session {session.id} — continuing (apply_plan will heal)")
    except Exception as exc:
        # Don't blow up the checkout — log loudly and keep going. apply_plan
        # will reconstruct the row from Stripe metadata when the user returns.
        print(f"[create_checkout] DB insert failed for session {session.id}: {type(exc).__name__}: {exc}")

    print(f"[create_checkout] DONE +{time.monotonic()-t0:.2f}s url={session.url[:80]}…")
    return {"url": session.url, "session_id": session.id}


async def apply_plan(session_id: str, session: Optional[dict] = None) -> dict:
    """Upgrade the user's plan based on a completed Stripe checkout session.

    Resilient to a missing payment_transactions row: if the DB insert at
    checkout-time was lost (or this server never saw it), we reconstruct
    the user_id and package_id from the Stripe session's metadata and
    UPSERT a row so subsequent polls can find it. This is the single most
    common reason "Hang tight" used to hang forever — apply_plan would
    raise 404 and the frontend silently swallowed it.
    """
    async with pool.acquire() as conn:
        tx = await conn.fetchrow(
            "SELECT * FROM payment_transactions WHERE session_id = $1", session_id,
        )

        # No DB row? Rebuild it from the Stripe session metadata so we can
        # still apply the plan — the user paid, they shouldn't be stuck.
        if not tx:
            if session is None:
                if not STRIPE_API_KEY:
                    raise HTTPException(503, "Stripe not configured")
                try:
                    session = stripe_lib().checkout.Session.retrieve(session_id)
                except Exception as exc:
                    print(f"[apply_plan] Stripe retrieve failed for {session_id}: {exc}")
                    raise HTTPException(502, f"Stripe retrieve failed: {exc}")
            # Normalise session to a plain dict — see _stripe_to_dict
            # docstring for why. metadata is a nested StripeObject too.
            sd = _stripe_to_dict(session)
            meta = _stripe_to_dict(sd.get("metadata"))
            user_id = meta.get("user_id")
            package_id = meta.get("package_id")
            if not user_id or not package_id:
                print(f"[apply_plan] Session {session_id} missing metadata.user_id or metadata.package_id; meta={meta}")
                raise HTTPException(400, "Stripe session missing user/package metadata")
            pkg_meta = PACKAGES.get(package_id)
            amount = (pkg_meta or {}).get("amount", 0.0)
            currency = (pkg_meta or {}).get("currency", "usd")
            await conn.execute(
                """INSERT INTO payment_transactions
                     (session_id, user_id, package_id, amount, currency, metadata, payment_status, status, created_at)
                   VALUES ($1,$2,$3,$4,$5,$6,'initiated','open',$7)
                   ON CONFLICT (session_id) DO NOTHING""",
                session_id, user_id, package_id, amount, currency,
                json.dumps({"reconstructed_from": "stripe_metadata"}), now_iso(),
            )
            tx = await conn.fetchrow(
                "SELECT * FROM payment_transactions WHERE session_id = $1", session_id,
            )
            print(f"[apply_plan] Reconstructed tx for session {session_id} (user={user_id}, pkg={package_id})")

        if tx["plan_applied"]:
            return dict(tx)

        pkg = PACKAGES.get(tx["package_id"])
        if not pkg:
            print(f"[apply_plan] Unknown package_id {tx['package_id']} on session {session_id}")
            return dict(tx)
        tier = pkg["tier"]
        expires = None
        if tier == "pro_weekly":
            expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        elif tier == "pro_monthly":
            expires = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        elif tier == "pro_yearly":
            expires = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        # Capture the Stripe customer + subscription IDs so we can later
        # open the billing portal for cancellation. session.customer is a
        # string id on most checkout sessions; subscription is set on
        # subscription-mode sessions only.
        cust_id = None
        sub_id = None
        if session is not None:
            sd2 = _stripe_to_dict(session)
            cust_id = sd2.get("customer")
            sub_id = sd2.get("subscription")
        # has_ever_subscribed sticks to TRUE the first time any Pro
        # tier applies — driver for the "no more first-time discount"
        # gate on the paywall. Never reset (even on cancellation), so
        # a churned-and-returning user pays the real price.
        if cust_id or sub_id:
            await conn.execute(
                """UPDATE users
                     SET plan_tier = $1, plan_expires_at = $2,
                         stripe_customer_id = COALESCE($3, stripe_customer_id),
                         stripe_subscription_id = COALESCE($4, stripe_subscription_id),
                         has_ever_subscribed = TRUE
                   WHERE user_id = $5""",
                tier, expires, cust_id, sub_id, tx["user_id"],
            )
        else:
            await conn.execute(
                """UPDATE users
                     SET plan_tier = $1, plan_expires_at = $2,
                         has_ever_subscribed = TRUE
                   WHERE user_id = $3""",
                tier, expires, tx["user_id"],
            )
        await conn.execute(
            """UPDATE payment_transactions SET plan_applied = TRUE,
                                                applied_at = $1,
                                                payment_status = 'paid',
                                                status = 'complete',
                                                updated_at = $1
               WHERE session_id = $2""",
            now_iso(), session_id,
        )
        print(f"[apply_plan] Applied {tier} to user {tx['user_id']} (session {session_id})")
        return dict(await conn.fetchrow(
            "SELECT * FROM payment_transactions WHERE session_id = $1", session_id,
        ))


# HTTPS bridge — Stripe sends users here after checkout, this page just
# deep-links them back into the app. No auth required (anyone with the
# session_id is fine; it's only useful to the app that created it).
@app.get("/api/payments/return")
async def payments_return(session_id: Optional[str] = None, action: str = "success"):
    from fastapi.responses import HTMLResponse
    deep_link = (
        f"overthink://payment-success?session_id={session_id}"
        if action == "success" and session_id
        else "overthink://paywall"
    )
    html = f"""<!doctype html>
<html><head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Returning to Overthink This…</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
            background: #F1ECE3; color: #161520;
            display: flex; align-items: center; justify-content: center;
            min-height: 100vh; margin: 0; padding: 0 24px; text-align: center; }}
    .card {{ max-width: 360px; }}
    h1 {{ font-size: 20px; margin: 0 0 8px; letter-spacing: -0.4px; }}
    p  {{ font-size: 14px; color: #5A586B; margin: 0 0 18px; }}
    a  {{ display: inline-block; padding: 12px 22px; background: #161520; color: #FFF;
          text-decoration: none; border-radius: 999px; font-weight: 700; font-size: 14px; }}
  </style>
  <script>
    window.location.replace({deep_link!r});
    setTimeout(function() {{ document.getElementById('manual').style.display = 'inline-block'; }}, 1500);
  </script>
</head><body>
  <div class="card">
    <h1>Returning you to Overthink This…</h1>
    <p>If the app doesn't open automatically, tap below.</p>
    <a id="manual" href={deep_link!r} style="display:none;">Open the app</a>
  </div>
</body></html>"""
    return HTMLResponse(content=html)


@app.get("/api/payments/status/{session_id}")
async def payment_status(session_id: str, user: dict = Depends(get_current_user)):
    """Polling endpoint the app hits while showing 'Verifying…' on the
    payment-success screen. Accepts either:
      • subscription checkout where session.status == 'complete'
      • one-time payment where payment_status == 'paid'
    Either way we apply the plan inline so the user doesn't need the webhook
    to fire first.

    Returns 200 with `applied` and `error` fields so the frontend can show
    real diagnostic info instead of silently spinning forever. We never
    raise from here — the frontend polls this URL repeatedly and a 5xx
    response would be silently swallowed.
    """
    if not STRIPE_API_KEY:
        return {
            "session_id": session_id, "applied": False,
            "error": "Stripe not configured on backend",
        }

    stripe = stripe_lib()
    session = None
    fetch_error: Optional[str] = None
    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except Exception as exc:
        fetch_error = f"Stripe retrieve failed: {exc}"
        print(f"[payment_status] {fetch_error} (session {session_id})")

    # Normalise to a plain dict — newer stripe-python (≥ 7.x) no longer
    # exposes .get() on StripeObject; accessing .get triggers __getattr__
    # which raises AttributeError. Going through dict() (or .to_dict() /
    # to_dict_recursive) gives us a vanilla dict we can interrogate
    # safely without per-version SDK guesswork.
    session_d = _stripe_to_dict(session) if session else {}
    payment_status_v = session_d.get("payment_status")
    sess_status      = session_d.get("status")
    mode             = session_d.get("mode")

    # A subscription is considered active once status == "complete" (Stripe
    # has charged the first invoice). A one-time payment uses payment_status.
    is_done = bool(session) and (
        payment_status_v == "paid"
        or payment_status_v == "no_payment_required"
        or (mode == "subscription" and sess_status == "complete")
    )

    apply_error: Optional[str] = None
    if is_done:
        try:
            await apply_plan(session_id, session=session)
        except HTTPException as exc:
            apply_error = f"apply_plan failed ({exc.status_code}): {exc.detail}"
            print(f"[payment_status] {apply_error} (session {session_id})")
        except Exception as exc:
            apply_error = f"apply_plan crashed: {exc}"
            print(f"[payment_status] {apply_error} (session {session_id})")

    # Read both the tx row AND the user's actual plan tier. Either being
    # "Pro" is enough for the frontend to stop polling — covers the case
    # where apply_plan succeeded on a previous request but the tx row was
    # never written for some reason.
    applied = False
    user_plan_tier: Optional[str] = None
    try:
        async with pool.acquire() as conn:
            tx = await conn.fetchrow(
                "SELECT plan_applied FROM payment_transactions WHERE session_id = $1",
                session_id,
            )
            row = await conn.fetchrow(
                "SELECT plan_tier FROM users WHERE user_id = $1", user["user_id"],
            )
        applied = bool(tx and tx["plan_applied"])
        user_plan_tier = row["plan_tier"] if row else None
    except Exception as exc:
        apply_error = (apply_error or "") + f" | DB read failed: {exc}"
        print(f"[payment_status] DB read failed for {session_id}: {exc}")

    # Cross-check: if the user is already on a paid tier, consider it done
    # regardless of DB transaction row state.
    user_is_pro = user_plan_tier in {"pro_weekly", "pro_monthly", "pro_yearly", "lifetime"}

    return {
        "session_id": session_id,
        "payment_status": payment_status_v,
        "status": sess_status,
        "mode": mode,
        "amount_total": session_d.get("amount_total"),
        "applied": applied or user_is_pro,
        "user_plan_tier": user_plan_tier,
        "error": fetch_error or apply_error,
    }


@app.post("/api/payments/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_API_KEY or not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(503, "Stripe not configured")
    stripe = stripe_lib()
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        raise HTTPException(400, f"Bad signature: {exc}")

    # Coerce the whole event to a plain dict — same reasoning as the
    # _stripe_to_dict helper. Lets us treat events from any stripe-python
    # version uniformly with [] / .get().
    event_d = _stripe_to_dict(event)
    et = event_d.get("type")
    data_obj = _stripe_to_dict(_stripe_to_dict(event_d.get("data")).get("object"))

    # Both events indicate "user paid" — checkout.session.completed fires
    # right after the user finishes checkout (subscription mode usually
    # has payment_status=paid by then; one-time always does). The async
    # invoice.payment_succeeded covers the rare case where the first
    # subscription invoice settles slightly after the session completes.
    # Subscription cancelled (manually or by failed payment) → drop the
    # user back to free. We look them up by stripe_subscription_id which we
    # stored at apply_plan time.
    if et == "customer.subscription.deleted":
        sub_id = data_obj.get("id")
        if sub_id:
            try:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """UPDATE users SET plan_tier = 'free', plan_expires_at = NULL,
                                            stripe_subscription_id = NULL
                           WHERE stripe_subscription_id = $1""",
                        sub_id,
                    )
                print(f"[webhook] subscription {sub_id} deleted → user reverted to free")
            except Exception as exc:
                print(f"[webhook] failed to revert user for sub {sub_id}: {exc}")
        return {"received": True}

    if et in {"checkout.session.completed", "checkout.session.async_payment_succeeded"}:
        session_d = data_obj  # already dict-coerced above
        mode = session_d.get("mode")
        pstatus = session_d.get("payment_status")
        sstatus = session_d.get("status")
        sid = session_d.get("id")
        is_paid = (
            pstatus == "paid"
            or pstatus == "no_payment_required"
            or (mode == "subscription" and sstatus == "complete")
        )
        if is_paid and sid:
            try:
                await apply_plan(sid, session=session_d)
            except Exception as exc:
                print(f"[webhook] apply_plan failed for {sid}: {exc}")
    return {"received": True}


@app.get("/api/payments/diag")
async def payments_diag():
    """Diagnostic endpoint — visible without auth so we can hit it from a
    browser to verify Stripe is reachable from this server. Returns the
    Stripe key prefix (NOT the full key) and the result of a no-op API
    call to confirm the key is valid for the current Stripe mode."""
    out: dict = {
        "stripe_key_present": bool(STRIPE_API_KEY),
        "stripe_key_prefix": (STRIPE_API_KEY[:8] + "…") if STRIPE_API_KEY else None,
        "stripe_key_is_test": STRIPE_API_KEY.startswith("sk_test_") if STRIPE_API_KEY else None,
        "stripe_key_is_live": STRIPE_API_KEY.startswith("sk_live_") if STRIPE_API_KEY else None,
        "webhook_secret_present": bool(STRIPE_WEBHOOK_SECRET),
        "db_pool_ready": pool is not None,
    }
    if not STRIPE_API_KEY:
        out["stripe_ok"] = False
        out["stripe_error"] = "No STRIPE_API_KEY env var on server"
        return out
    try:
        stripe = stripe_lib()
        # Cheapest possible call: just list 1 product to confirm the key
        # authenticates against the right Stripe account.
        stripe.Product.list(limit=1)
        out["stripe_ok"] = True
    except Exception as exc:
        out["stripe_ok"] = False
        out["stripe_error"] = f"{type(exc).__name__}: {exc}"
    return out


@app.post("/api/payments/portal")
async def billing_portal(user: dict = Depends(get_current_user)):
    """Create a Stripe Billing Portal session — the user gets a URL where
    they can manage / cancel their subscription. We pre-stored the
    customer_id at apply_plan time. Returns 400 if the user has no
    associated Stripe customer (e.g. they're free or on a guest account)."""
    if user.get("is_guest"):
        raise HTTPException(403, "Sign in to manage your subscription")
    if not STRIPE_API_KEY:
        raise HTTPException(503, "Stripe not configured")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT stripe_customer_id, plan_tier FROM users WHERE user_id = $1",
            user["user_id"],
        )
    cust_id = row["stripe_customer_id"] if row else None
    if not cust_id:
        raise HTTPException(400, "No subscription on file")
    # The portal redirects back here when the user is done. Routes through
    # the HTTPS bridge → app deep link the same way checkout does.
    base_https = "https://overthink-this-api.onrender.com"
    return_url = f"{base_https}/api/payments/return?action=cancel"
    try:
        stripe = stripe_lib()
        portal = stripe.billing_portal.Session.create(
            customer=cust_id,
            return_url=return_url,
        )
    except Exception as exc:
        detail = getattr(exc, "user_message", None) or str(exc)
        print(f"[billing_portal] Stripe rejected portal create: {type(exc).__name__}: {detail}")
        raise HTTPException(502, f"Stripe error: {detail}")
    return {"url": portal.url}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"ok": True, "db": pool is not None, "ai": bool(GEMINI_API_KEY), "stripe": bool(STRIPE_API_KEY)}
