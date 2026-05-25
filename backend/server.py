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
from typing import Any, Optional

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
        # Idempotent column adds for fields added after the initial schema.
        # Safe to re-run on every boot — IF NOT EXISTS makes it a no-op once
        # the column is in place. Avoids needing a manual migration step.
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT"
                )
                await conn.execute(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_subscription_id TEXT"
                )
                # Onboarding step 2: personality + spiral-frequency self-report.
                # bio is a free-form text the user writes about themselves; the
                # other two are short canned tags from the onboarding picker.
                # All optional — nothing reads them as required.
                await conn.execute(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS bio TEXT"
                )
                await conn.execute(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS personality TEXT"
                )
                await conn.execute(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS spiral_frequency TEXT"
                )
        except Exception as exc:
            print(f"[lifespan] column-add migration warning: {exc}")
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
        "streak_count": u.get("streak_count") or 0,
        "xp": u.get("xp") or 0,
        "level": u.get("level") or 1,
        "phone_number": u.get("phone_number"),
        "phone_verified": bool(u.get("phone_verified")),
        "customization": customization,
        "unlocked_items": unlocked,
        "bio": u.get("bio") or None,
        "personality": u.get("personality") or None,
        "spiral_frequency": u.get("spiral_frequency") or None,
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
- The verdict MUST be a punchline. It should make someone reading over their shoulder laugh out loud. Examples: "You've been workshopping a tragedy for an audience of one." / "Congratulations, you're the lead in a one-act play nobody wrote." / "Sara is brushing her teeth. You're writing her closing argument."
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
situation: "I sent a long message to Sara at 1am about how I feel and she hasn't replied in 6 hours. I'm sure she's losing interest and I shouldn't have said anything. I'm about to send a follow-up apologizing."

EXAMPLE OUTPUT:
{
  "outcomes": [
    {
      "title": "Sara is asleep, then busy",
      "description": "It is currently 6 hours since you sent a long emotional message at 1am. The simplest read here is that Sara slept through it, woke up, had her morning, and hasn't gotten back to her phone yet. People don't sit on the couch refreshing their inbox waiting to engineer the perfect emotional response — they live their day. The most boring explanation is also the one that doesn't require you to be a problem. You'll probably hear back this afternoon and it'll be normal.",
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
    "When Sara picks up her phone",
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
    "verdict_text": "Sara is having a Tuesday. You're having a trial."
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
    into the tone-specific template."""
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
  "name": "1-2 words. A short label for this spiral, like a chapter title. Capitalised. Example: 'Sara Silence', 'Boss Meeting', 'Late Text', 'The Apartment'.",
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
    "verdict_text": "A MOTTO. Maximum 12 words. Tattoo-worthy. Hits like a punchline. Examples: 'You're auditioning for a role nobody's casting.' / 'Sara is having a Tuesday. You're having a trial.'"
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


async def run_gemini(situation_text: str, tone: str, category: str = "") -> dict:
    if not GEMINI_API_KEY:
        print("[gemini] ❌ no GEMINI_API_KEY in env — using fallback")
        return _fallback_with_tag(situation_text, tone, "no_api_key")
    try:
        # NEW SDK — `pip install google-genai` (NOT the deprecated google-generativeai)
        from google import genai
        from google.genai import types as gen_types

        client = genai.Client(api_key=GEMINI_API_KEY)

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

        text = ""
        used_model = None
        last_error: Optional[Exception] = None
        for model_name in GEMINI_MODELS:
            try:
                print(f"[gemini]   trying {model_name}…")
                resp = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                text = (resp.text or "").strip()
                if text:
                    used_model = model_name
                    print(f"[gemini] ✅ {model_name} returned {len(text)} chars")
                    break
                else:
                    print(f"[gemini]   {model_name} returned empty text, trying next…")
            except Exception as me:
                last_error = me
                print(f"[gemini]   {model_name} failed: {type(me).__name__}: {str(me)[:160]}")
                continue

        if not text:
            raise last_error or ValueError("All Gemini models failed to respond")

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
    if not GEMINI_API_KEY:
        print("[text-check] no GEMINI_API_KEY — using fallback")
        return _text_check_fallback("no_api_key")
    try:
        from google import genai
        from google.genai import types as gen_types
        client = genai.Client(api_key=GEMINI_API_KEY)
        config = gen_types.GenerateContentConfig(
            temperature=1.05,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )
        prompt = _build_text_check_prompt(draft=draft, context=context, relationship=relationship)
        text = ""
        last_error: Optional[Exception] = None
        for model_name in GEMINI_MODELS:
            try:
                print(f"[text-check] trying {model_name}…")
                resp = await client.aio.models.generate_content(
                    model=model_name, contents=prompt, config=config,
                )
                text = (resp.text or "").strip()
                if text:
                    print(f"[text-check] ✅ {model_name} returned {len(text)} chars")
                    break
            except Exception as me:
                last_error = me
                print(f"[text-check] {model_name} failed: {type(me).__name__}: {str(me)[:160]}")
                continue
        if not text:
            raise last_error or ValueError("All Gemini models failed")
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


@app.post("/api/text-check")
async def text_check(body: TextCheckRequest, user: dict = Depends(get_current_user)):
    """Pre-send draft analyzer. Returns three predicted responses + a
    verdict + optional rewrite. Free tier capped at FREE_TEXT_CHECK_MONTHLY
    per calendar month (UTC); Pro is unlimited."""
    global _TEXT_CHECK_COLUMNS_READY
    db_required()
    if user.get("is_guest"):
        raise HTTPException(403, "Sign in to use Text-Check")
    draft = (body.draft or "").strip()
    if not draft:
        raise HTTPException(400, "Draft is empty")
    if len(draft) > 4000:
        raise HTTPException(400, "Draft is too long (4000 char max)")

    is_pro_user = (user.get("plan_tier") or "free") in {"pro_weekly", "pro_monthly", "pro_yearly", "lifetime"}
    if not _TEXT_CHECK_COLUMNS_READY:
        try:
            await _ensure_text_check_columns()
            _TEXT_CHECK_COLUMNS_READY = True
        except Exception as exc:
            print(f"[text-check] column-ensure failed (non-fatal): {exc}")

    this_month = datetime.now(timezone.utc).strftime("%Y-%m")
    used_this_month = 0
    if not is_pro_user:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT text_checks_month, text_checks_count FROM users WHERE user_id = $1",
                user["user_id"],
            )
        if row and row["text_checks_month"] == this_month:
            used_this_month = row["text_checks_count"] or 0
        if used_this_month >= FREE_TEXT_CHECK_MONTHLY:
            raise HTTPException(
                402,
                f"Free tier limit reached ({FREE_TEXT_CHECK_MONTHLY}/month). Upgrade for unlimited.",
            )

    result = await run_text_check(
        draft=draft,
        context=body.context or "",
        relationship=body.relationship or "someone",
    )

    # Bump the counter only on a successful AI call (not fallback) so a
    # backend outage doesn't burn the user's free quota.
    if not is_pro_user and result.get("_ai_source") == "live":
        async with pool.acquire() as conn:
            await conn.execute(
                """UPDATE users
                     SET text_checks_month = $1,
                         text_checks_count = CASE
                            WHEN text_checks_month = $1 THEN COALESCE(text_checks_count, 0) + 1
                            ELSE 1
                         END
                   WHERE user_id = $2""",
                this_month, user["user_id"],
            )
        used_this_month += 1

    return {
        "result": result,
        "usage": {
            "is_pro": is_pro_user,
            "used_this_month": used_this_month,
            "monthly_limit": None if is_pro_user else FREE_TEXT_CHECK_MONTHLY,
            "remaining": None if is_pro_user else max(0, FREE_TEXT_CHECK_MONTHLY - used_this_month),
        },
    }


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


# ---------------------------------------------------------------------------
# Streak handling
# ---------------------------------------------------------------------------

async def bump_streak_and_total(user: dict) -> dict:
    today = today_iso_date()
    last = user.get("last_active")
    last_date = None
    try:
        if last:
            last_date = datetime.fromisoformat(last.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        last_date = None

    streak = user.get("streak_count") or 0
    if last_date == today:
        new_streak = streak or 1
    else:
        # Was last_date == yesterday?
        try:
            yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        except Exception:
            yesterday = None
        if last_date == yesterday:
            new_streak = streak + 1
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
                spirals_used_today = $3,
                spirals_used_date = $4,
                spirals_total = spirals_total + 1
               WHERE user_id = $5""",
            new_streak, now_iso(), used_today, today, user["user_id"],
        )
        return dict(await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user["user_id"]))


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

        async with pool.acquire() as conn:
            await conn.execute(
                """UPDATE spirals SET
                    outcomes = $1,
                    verdict = $2,
                    status = 'complete',
                    error_message = $3,
                    name = COALESCE(name, $5)
                   WHERE id = $4""",
                json.dumps(payload.get("outcomes", [])),
                json.dumps(payload.get("verdict", {})),
                f"fallback: {ai_reason}" if ai_source == "fallback" else "live",
                spiral_id,
                # Gemini's auto-name, trimmed and capped at 32 chars
                (payload.get("name") or "Untitled").strip()[:32],
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
    # Compute pattern_context on the fly — finds past spirals in the
    # same category with ≥ 0.34 Jaccard tag overlap (same heuristic as
    # /api/insights/loops). Cheap enough for a single-spiral GET that
    # we don't bother caching it. Recomputed on every read so the
    # "last resolution" stays fresh as the user resolves more spirals.
    spiral["pattern_context"] = await _compute_pattern_context(
        user_id=user["user_id"],
        current_spiral=dict(row),
    )
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
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT f.id, f.name, f.color, f.created_at,
                      COUNT(s.id) AS spiral_count
                 FROM folders f
                 LEFT JOIN spirals s
                        ON s.folder_id = f.id AND s.user_id = f.user_id
                WHERE f.user_id = $1
                GROUP BY f.id
                ORDER BY f.created_at DESC""",
            user["user_id"],
        )
    return {"folders": [dict(r) for r in rows]}


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
        # Detach any spirals from the deleted folder (don't delete them)
        await conn.execute(
            "UPDATE spirals SET folder_id = NULL WHERE folder_id = $1 AND user_id = $2",
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
    return {"ok": True}


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
# endpoint surfaces "you've spiralled about Sara 7 times this month — 6
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
    # Sort by count desc, then by recency of the most recent spiral.
    def _sort_key(cl):
        max_dt = max((s.get("created_at") or "" for s in cl), default="")
        return (-len(cl), -ord(max_dt[0]) if max_dt else 0)
    clusters.sort(key=lambda cl: (-len(cl), -(len(cl[0].get("created_at") or ""))))
    loops = [_summarise_loop(cl) for cl in clusters[:8]]
    return {
        "loops": loops,
        "min_spirals_needed": 3,
        "total_spirals": len(spirals),
        "is_pro": is_pro_user,
        # The frontend uses this to decide whether to blur loops [1:].
        # We don't strip the data server-side because the user owns it —
        # gating is purely visual / UX.
        "preview_only": not is_pro_user,
    }


# ---------------------------------------------------------------------------
# Payments — Stripe Checkout
# ---------------------------------------------------------------------------

PACKAGES = {
    "weekly":  {"id": "weekly",  "label": "Spiral Starter",      "amount": 1.99,  "currency": "usd", "tier": "pro_weekly"},
    "monthly": {"id": "monthly", "label": "Deep In My Feelings", "amount": 5.99,  "currency": "usd", "tier": "pro_monthly"},
    # Annual plan — billed once per year, treated as Pro for the whole
    # year. Tier "pro_yearly" so isPro() in the frontend keeps working.
    # Apply_plan sets plan_expires_at to now+365d on activation.
    "yearly":  {"id": "yearly",  "label": "Year of Quiet",       "amount": 39.99, "currency": "usd", "tier": "pro_yearly"},
    "lifetime":{"id": "lifetime","label": "Infinite Loop Pass",  "amount": 29.99, "currency": "usd", "tier": "lifetime"},
}


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
        if cust_id or sub_id:
            await conn.execute(
                """UPDATE users
                     SET plan_tier = $1, plan_expires_at = $2,
                         stripe_customer_id = COALESCE($3, stripe_customer_id),
                         stripe_subscription_id = COALESCE($4, stripe_subscription_id)
                   WHERE user_id = $5""",
                tier, expires, cust_id, sub_id, tx["user_id"],
            )
        else:
            await conn.execute(
                "UPDATE users SET plan_tier = $1, plan_expires_at = $2 WHERE user_id = $3",
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
