// Supabase Edge Function: "api"
// ---------------------------------------------------------------------------
// Migration Phase 2 — the data layer. Ports the read/write CRUD endpoints
// off Render so the archive, spiral lists, folders, etc. load fast (no
// Render cold start). Action-based router: POST { action, ...params }.
// Authenticates with the EXISTING custom session token via the sessions
// table (service role) — no auth rewrite. Per-user access is enforced in
// code (every query is scoped to the authenticated user_id), exactly like
// server.py did.
// ---------------------------------------------------------------------------

import { createClient } from "jsr:@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SERVICE_ROLE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const db = createClient(SUPABASE_URL, SERVICE_ROLE, { auth: { persistSession: false } });

const PRO_TIERS = new Set(["pro_weekly", "pro_monthly", "pro_yearly", "lifetime"]);
const isPro = (t?: string) => PRO_TIERS.has(t ?? "free");

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-app-key, content-type, accept",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...CORS } });
}
const nowIso = () => new Date().toISOString();
const todayDate = () => new Date().toISOString().slice(0, 10);
function rid(prefix: string) { return `${prefix}_${crypto.randomUUID().replace(/-/g, "").slice(0, 14)}`; }

async function getUser(req: Request): Promise<any | null> {
  const auth = req.headers.get("authorization") ?? "";
  const token = auth.toLowerCase().startsWith("bearer ") ? auth.slice(7).trim() : "";
  if (!token) return null;
  const { data: sess } = await db.from("sessions").select("user_id").eq("session_token", token).maybeSingle();
  if (!sess) return null;
  const { data: user } = await db.from("users").select("*").eq("user_id", sess.user_id).maybeSingle();
  return user ?? null;
}

// ── serializers (mirror server.py *_public) ─────────────────────────────
function userPublic(u: any) {
  return {
    user_id: u.user_id, email: u.email ?? null, name: u.name,
    picture: u.picture ?? null, is_guest: !!u.is_guest,
    tone_preference: u.tone_preference ?? "balanced",
    default_category: u.default_category ?? "other",
    plan_tier: u.plan_tier ?? "free",
    spirals_used_today: u.spirals_used_today ?? 0,
    spirals_total: u.spirals_total ?? 0,
    streak_count: u.streak_count ?? 0,
    xp: u.xp ?? 0, level: u.level ?? 1,
    phone_number: u.phone_number ?? null, phone_verified: !!u.phone_verified,
    customization: u.customization ?? null,
    unlocked_items: u.unlocked_items ?? [],
    bio: u.bio ?? null, personality: u.personality ?? null, spiral_frequency: u.spiral_frequency ?? null,
    has_ever_subscribed: !!u.has_ever_subscribed,
    streak_freezes_remaining: u.streak_freezes_remaining ?? 3,
    streak_freezes_max: 3,
    created_at: u.created_at,
  };
}

// ── action handlers ─────────────────────────────────────────────────────
// ── XP / level curve (port of server.py) ────────────────────────────────
function xpForLevel(level: number): number {
  if (level < 1) return 200;
  if (level <= 5) return 200;
  if (level <= 10) return 500;
  if (level <= 20) return 1200;
  if (level <= 35) return 3000;
  if (level <= 50) return 6500;
  if (level <= 70) return 12000;
  if (level <= 85) return 22000;
  return 38000;
}
function totalXpForLevel(level: number): number {
  let t = 0;
  for (let lv = 1; lv < level; lv++) t += xpForLevel(lv);
  return t;
}
function levelFromXp(xp: number): number {
  let lv = 1;
  while (lv < 100 && totalXpForLevel(lv + 1) <= xp) lv++;
  return lv;
}
const LEVEL_UNLOCKS: [number, string, string, string][] = [
  [5, "title", "The Apprentice", "Title: The Apprentice"],
  [10, "name_color", "#C8A0DC", "Lilac name colour"],
  [15, "card_theme", "midnight", "Midnight share-card theme"],
  [20, "title", "The Worrier", "Title: The Worrier"],
  [25, "name_color", "#F5C518", "Gold name colour"],
  [30, "card_theme", "warm", "Warm Ember share-card theme"],
  [35, "title", "The Analyst", "Title: The Analyst"],
  [40, "name_color", "#4FC3F7", "Ocean name colour"],
  [45, "card_theme", "forest", "Deep Forest share-card theme"],
  [50, "title", "Spiral Veteran", "Title: Spiral Veteran"],
  [55, "name_color", "#E24B4A", "Crimson name colour"],
  [60, "card_theme", "aurora", "Aurora Borealis share-card theme"],
  [65, "title", "The Philosopher", "Title: The Philosopher"],
  [70, "name_color", "#34A56F", "Moss name colour"],
  [75, "title", "The Oracle", "Title: The Oracle"],
  [80, "name_color", "#FF8C42", "Ember name colour"],
  [85, "card_theme", "neon", "Neon Noir share-card theme"],
  [90, "title", "Mind Cartographer", "Title: Mind Cartographer"],
  [95, "title", "Overthinking Champion", "Title: Overthinking Champion"],
  [100, "title", "Overthinker Supreme", "Title: Overthinker Supreme"],
];

async function createSession(userId: string): Promise<string> {
  const token = (crypto.randomUUID() + crypto.randomUUID()).replace(/-/g, "");
  const expires = new Date(Date.now() + 30 * 24 * 3600 * 1000).toISOString();
  await db.from("sessions").insert({ session_token: token, user_id: userId, expires_at: expires, created_at: nowIso() });
  return token;
}

function unlocksForLevel(level: number): string[] {
  const out: string[] = [];
  for (const [lv, kind, value] of LEVEL_UNLOCKS) if (level >= lv) out.push(`${kind}:${value}`);
  return out;
}
async function grantXp(userId: string, amount: number): Promise<any> {
  const { data: row } = await db.from("users").select("xp").eq("user_id", userId).single();
  const newXp = (row?.xp ?? 0) + amount;
  const newLevel = levelFromXp(newXp);
  await db.from("users").update({ xp: newXp, level: newLevel, unlocked_items: unlocksForLevel(newLevel) }).eq("user_id", userId);
  const { data: u } = await db.from("users").select("*").eq("user_id", userId).single();
  return u;
}

function weekPeriod(): string {
  const d = new Date();
  const day = (d.getUTCDay() + 6) % 7; // Mon=0
  d.setUTCDate(d.getUTCDate() - day);
  return `W${d.toISOString().slice(0, 10)}`;
}

const DAILY_TASKS = [
  { task_id: "first_spiral", label: "Create your first spiral today", xp: 50, target: 1 },
  { task_id: "resolve_one", label: "Resolve one spiral", xp: 75, target: 1 },
  { task_id: "brutal_tone", label: "Use the Brutal tone", xp: 40, target: 1 },
  { task_id: "long_spiral", label: "Write a 100+ word spiral", xp: 60, target: 1 },
  { task_id: "share_verdict", label: "Share a verdict", xp: 40, target: 1 },
  { task_id: "plot_twist", label: "Log a plot-twist resolution", xp: 80, target: 1 },
  { task_id: "one_text_check", label: "Stop one text from going out", xp: 60, target: 1 },
  { task_id: "one_compat", label: "Run a compatibility test", xp: 60, target: 1 },
];
const WEEKLY_TASKS = [
  { task_id: "seven_spirals", label: "Create 7 spirals this week", xp: 500, target: 7 },
  { task_id: "five_resolved", label: "Resolve 5 spirals", xp: 400, target: 5 },
  { task_id: "all_tones", label: "Use all 3 tones", xp: 300, target: 3 },
  { task_id: "five_streak", label: "Hit a 5-day streak", xp: 350, target: 5 },
  { task_id: "four_categories", label: "4 different categories", xp: 450, target: 4 },
  { task_id: "share_three", label: "Share 3 verdicts", xp: 250, target: 3 },
  { task_id: "gentle_three", label: "Use the Gentle tone 3x", xp: 180, target: 3 },
  { task_id: "three_streak", label: "Hit a 3-day streak", xp: 200, target: 3 },
  { task_id: "three_text_checks", label: "Run 3 text-checks", xp: 280, target: 3 },
  { task_id: "two_compats", label: "Run 2 compatibility tests", xp: 240, target: 2 },
  { task_id: "all_three_kinds", label: "Use all 3 tools (spiral / text / compat)", xp: 350, target: 3 },
];
async function ensureTaskRows(userId: string, day: string, week: string) {
  for (const t of DAILY_TASKS) {
    await db.from("user_tasks").upsert(
      { user_id: userId, task_id: t.task_id, period: day, target: t.target, progress: 0, created_at: nowIso() },
      { onConflict: "user_id,task_id,period", ignoreDuplicates: true },
    );
  }
  for (const t of WEEKLY_TASKS) {
    await db.from("user_tasks").upsert(
      { user_id: userId, task_id: t.task_id, period: week, target: t.target, progress: 0, created_at: nowIso() },
      { onConflict: "user_id,task_id,period", ignoreDuplicates: true },
    );
  }
}

const DROPS = [
  { id: "stoic", label: "The Stoic", sub: "Marcus Aurelius energy", icon: "shield-outline", start: "2025-01-01", end: "2099-01-01" },
];

// Insights score — folds spirals + text-checks + compat resolutions into
// one "how wrong your brain was" tally (port of server.py).
async function insightsScore(user: any): Promise<Response> {
  const [sp, tc, cp] = await Promise.all([
    db.from("spirals").select("category,tone_used,resolution_status").eq("user_id", user.user_id).eq("status", "complete"),
    db.from("text_checks").select("resolution_status").eq("user_id", user.user_id).not("resolution_status", "is", null),
    db.from("compatibility_tests").select("resolution_status").eq("user_id", user.user_id).not("resolution_status", "is", null),
  ]);
  const items: any[] = [];
  for (const r of sp.data ?? []) items.push({ category: r.category, tone: r.tone_used, rs: r.resolution_status });
  for (const r of tc.data ?? []) items.push({ category: "text_check", tone: "balanced", rs: r.resolution_status });
  for (const r of cp.data ?? []) items.push({ category: "compatibility", tone: "balanced", rs: r.resolution_status });
  let right = 0, wrong = 0, plot = 0, resolvedCount = 0;
  const breakdown: Record<string, number> = {};
  const catRight: Record<string, number> = {}, catWrong: Record<string, number> = {};
  for (const it of items) {
    if (!it.rs) continue;
    resolvedCount++; breakdown[it.rs] = (breakdown[it.rs] ?? 0) + 1;
    const isRight = it.rs === "resolved" || it.rs === "plot_twist_good";
    const isWrong = it.rs === "not_resolved" || it.rs === "plot_twist_bad";
    if (it.rs === "plot_twist_good" || it.rs === "plot_twist_bad") plot++;
    if (isRight) { right++; catRight[it.category] = (catRight[it.category] ?? 0) + 1; }
    if (isWrong) { wrong++; catWrong[it.category] = (catWrong[it.category] ?? 0) + 1; }
  }
  const decisive = right + wrong;
  const pct = decisive > 0 ? Math.round((right / decisive) * 100) : null;
  const share = pct === null ? "Still tracking — resolve more to see your number."
    : pct >= 70 ? `My brain was wrong ${pct}% of the time.`
    : pct >= 50 ? `My worst case missed more than it landed (${pct}%).`
    : `My brain was actually right ${100 - pct}% of the time. Rude.`;
  return json({
    total_spirals: items.length, resolved_count: resolvedCount,
    worst_case_avoided_pct: pct, resolution_breakdown: breakdown,
    best_tone: null, worst_category: null, best_category: null,
    plot_twist_count: plot, share_line: share, is_pro: isPro(user.plan_tier), preview_only: false,
  });
}

// Insights loops — cluster the archive by category (simplified port).
async function insightsLoops(user: any): Promise<Response> {
  const { data: rows } = await db.from("spirals").select("category,tags,resolution_status,resolved,created_at,situation_text")
    .eq("user_id", user.user_id).eq("status", "complete");
  const byCat: Record<string, any[]> = {};
  for (const r of rows ?? []) (byCat[r.category] ??= []).push(r);
  const loops = Object.entries(byCat)
    .filter(([, arr]) => arr.length >= 2)
    .map(([cat, arr]) => {
      const resolvedCount = arr.filter((s) => s.resolution_status).length;
      return {
        headline: `Your ${cat} loop`, category: cat, count: arr.length,
        tags: [], last_seen: arr[0]?.created_at ?? null,
        resolution_breakdown: {}, resolved_count: resolvedCount,
        worry_accuracy_pct: null,
        sample_situations: arr.slice(0, 3).map((s) => (s.situation_text ?? "").slice(0, 100)),
      };
    })
    .sort((a, b) => b.count - a.count);
  return json({ loops, min_spirals_needed: 2, total_spirals: rows?.length ?? 0, is_pro: isPro(user.plan_tier), preview_only: !isPro(user.plan_tier) });
}

type Ctx = { user: any; body: any; req: Request };

// Actions callable WITHOUT a session token (login).
const UNAUTHED = new Set(["auth.guest", "auth.google"]);

const handlers: Record<string, (c: Ctx) => Promise<Response>> = {
  // ---- auth / login ----
  "auth.guest": async ({ body, req }) => {
    const ip = (req.headers.get("x-forwarded-for") ?? "").split(",")[0].trim() || "0.0.0.0";
    const cutoff = new Date(Date.now() - 24 * 3600 * 1000).toISOString();
    const { count } = await db.from("users").select("user_id", { count: "exact", head: true })
      .eq("is_guest", true).eq("ip_address", ip).gt("created_at", cutoff);
    if ((count ?? 0) >= 5) return json({ detail: "Too many guest accounts from this IP. Please sign in instead." }, 429);
    const uid = `guest_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
    await db.from("users").insert({
      user_id: uid, name: body.name || "Guest", is_guest: true, plan_tier: "free",
      created_at: nowIso(), ip_address: ip, last_active: nowIso(), customization: {}, unlocked_items: [],
    });
    const { data: u } = await db.from("users").select("*").eq("user_id", uid).single();
    const token = await createSession(uid);
    return json({ user: userPublic(u), session_token: token });
  },
  "auth.google": async ({ body, req }) => {
    let info: any;
    try {
      const r = await fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
        headers: { Authorization: `Bearer ${body.access_token}` },
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      info = await r.json();
    } catch (e) {
      return json({ detail: `Google auth failed: ${e}` }, 401);
    }
    const email = info.email;
    if (!email) return json({ detail: "Google profile missing email" }, 401);
    const name = info.name || String(email).split("@")[0];
    const picture = info.picture ?? null;
    const ip = (req.headers.get("x-forwarded-for") ?? "").split(",")[0].trim() || null;
    const newId = `user_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
    const { data: existing } = await db.from("users").select("*").eq("email", email).maybeSingle();
    let u = existing;
    if (existing) {
      await db.from("users").update({ name, picture, last_active: nowIso() }).eq("email", email);
      const { data } = await db.from("users").select("*").eq("email", email).single();
      u = data;
    } else {
      await db.from("users").insert({
        user_id: newId, email, name, picture, is_guest: false, plan_tier: "free",
        created_at: nowIso(), ip_address: ip, last_active: nowIso(), customization: {}, unlocked_items: [],
      });
      const { data } = await db.from("users").select("*").eq("email", email).single();
      u = data;
    }
    const token = await createSession(u.user_id);
    return json({ user: userPublic(u), session_token: token });
  },
  "auth.logout": async ({ req }) => {
    const auth = req.headers.get("authorization") ?? "";
    const token = auth.toLowerCase().startsWith("bearer ") ? auth.slice(7).trim() : "";
    if (token) await db.from("sessions").delete().eq("session_token", token);
    return json({ ok: true });
  },

  // ---- level / XP ----
  "level": async ({ user }) => {
    const xp = user.xp ?? 0;
    const level = levelFromXp(xp);
    const startXp = totalXpForLevel(level);
    const endXp = level < 100 ? totalXpForLevel(level + 1) : startXp;
    const xpInLevel = xp - startXp;
    const xpNeeded = Math.max(endXp - startXp, 1);
    const pct = Math.round(Math.min(xpInLevel / xpNeeded, 1) * 100);
    let unlocked = user.unlocked_items ?? [];
    if (typeof unlocked === "string") { try { unlocked = JSON.parse(unlocked); } catch { unlocked = []; } }
    const upcoming = [];
    for (const [lv, kind, value, label] of LEVEL_UNLOCKS) {
      if (lv > level) upcoming.push({ level: lv, kind, value, label });
      if (upcoming.length >= 5) break;
    }
    return json({
      level, xp, xp_in_level: xpInLevel, xp_needed: xpNeeded, pct,
      unlocked_items: unlocked, upcoming_unlocks: upcoming,
      all_unlocks: LEVEL_UNLOCKS.map(([lv, k, v, label]) => ({ level: lv, kind: k, value: v, label })),
    });
  },

  // ---- activity buckets ----
  "activity": async ({ user, body }) => {
    const range = body.range_ ?? "week";
    const days = range === "year" ? 365 : range === "month" ? 30 : 7;
    const start = new Date(Date.now() - days * 24 * 3600 * 1000).toISOString().slice(0, 10);
    const { data: rows } = await db.from("spirals").select("created_at").eq("user_id", user.user_id).gte("created_at", start);
    const buckets: Record<string, number> = {};
    for (const r of rows ?? []) {
      const d = String(r.created_at).slice(0, 10);
      const key = range === "year" ? d.slice(0, 7) : d;
      buckets[key] = (buckets[key] ?? 0) + 1;
    }
    return json({ range, buckets: Object.entries(buckets).sort().map(([date, count]) => ({ date, count })) });
  },

  // ---- tasks / XP ----
  "tasks.get": async ({ user }) => {
    if (user.is_guest || (user.plan_tier ?? "free") === "free") return json({ detail: "Pro required for tasks" }, 402);
    const day = todayDate();
    const week = weekPeriod();
    await ensureTaskRows(user.user_id, day, week);
    const { data: rows } = await db.from("user_tasks").select("task_id,period,progress,target,claimed")
      .eq("user_id", user.user_id).in("period", [day, week]);
    const byId: Record<string, any> = {};
    for (const r of rows ?? []) byId[`${r.task_id}|${r.period}`] = r;
    const build = (list: any[], period: string) => list.map((t) => {
      const row = byId[`${t.task_id}|${period}`];
      return { ...t, period, progress: row?.progress ?? 0, claimed: row?.claimed ?? false };
    });
    return json({ daily: build(DAILY_TASKS, day), weekly: build(WEEKLY_TASKS, week) });
  },
  "tasks.claim": async ({ user, body }) => {
    const day = todayDate(); const week = weekPeriod();
    const { data: row } = await db.from("user_tasks").select("*").eq("user_id", user.user_id)
      .eq("task_id", body.task_id).in("period", [day, week]).maybeSingle();
    if (!row) return json({ detail: "Task not found" }, 404);
    if (row.claimed) return json({ detail: "Already claimed" }, 400);
    if (row.progress < row.target) return json({ detail: "Task not complete" }, 400);
    const all: Record<string, any> = {}; for (const t of [...DAILY_TASKS, ...WEEKLY_TASKS]) all[t.task_id] = t;
    const xp = all[body.task_id]?.xp ?? 0;
    await db.from("user_tasks").update({ claimed: true, claimed_at: nowIso() }).eq("id", row.id);
    const nu = await grantXp(user.user_id, xp);
    return json({ claimed: true, xp_granted: xp, level: nu.level, xp: nu.xp, unlocked_items: nu.unlocked_items ?? [] });
  },

  // ---- drops ----
  "drops.current": async ({ user }) => {
    const today = todayDate();
    const active = DROPS.find((d) => d.start <= today && today < d.end) ?? null;
    const pub = (d: any) => ({ id: d.id, label: d.label, sub: d.sub, icon: d.icon, starts_at: d.start, ends_at: d.end, tone_id: `drop:${d.id}` });
    return json({ active: active ? pub(active) : null, next: null, is_pro: isPro(user.plan_tier) });
  },

  // ---- profile stats ----
  "profile.stats": async ({ user }) => {
    const { data: rows } = await db.from("spirals").select("category,tone_used,resolved").eq("user_id", user.user_id);
    const total = rows?.length ?? 0;
    const resolved = (rows ?? []).filter((r: any) => r.resolved).length;
    const byCat: Record<string, number> = {}, byTone: Record<string, number> = {};
    for (const r of rows ?? []) { byCat[r.category] = (byCat[r.category] ?? 0) + 1; byTone[r.tone_used] = (byTone[r.tone_used] ?? 0) + 1; }
    return json({ total, resolved, resolved_pct: total ? Math.round((resolved / total) * 100) : 0, by_category: byCat, by_tone: byTone });
  },
  "profile.full_stats": async ({ user }) => {
    const { data: rows } = await db.from("spirals").select("category,tone_used,resolved,resolution_status,created_at").eq("user_id", user.user_id);
    const total = rows?.length ?? 0;
    const resolved = (rows ?? []).filter((r: any) => r.resolved).length;
    const byCat: Record<string, number> = {}, byTone: Record<string, number> = {};
    for (const r of rows ?? []) { byCat[r.category] = (byCat[r.category] ?? 0) + 1; byTone[r.tone_used] = (byTone[r.tone_used] ?? 0) + 1; }
    return json({
      total, resolved, resolved_pct: total ? Math.round((resolved / total) * 100) : 0,
      by_category: byCat, by_tone: byTone,
      streak_count: user.streak_count ?? 0, xp: user.xp ?? 0, level: levelFromXp(user.xp ?? 0),
      spirals_total: user.spirals_total ?? 0,
    });
  },
  "wrapped": async ({ user }) => {
    const { data: rows } = await db.from("spirals").select("category").eq("user_id", user.user_id);
    const byCat: Record<string, number> = {};
    for (const r of rows ?? []) byCat[r.category] = (byCat[r.category] ?? 0) + 1;
    const top = Object.entries(byCat).sort((a, b) => b[1] - a[1])[0];
    return json({ total_spirals: rows?.length ?? 0, top_category: top?.[0] ?? null, streak_count: user.streak_count ?? 0 });
  },

  // ---- insights ----
  "insights.score": async ({ user }) => insightsScore(user),
  "insights.loops": async ({ user }) => insightsLoops(user),
  "insights.checkin_candidate": async ({ user }) => {
    const { data } = await db.from("spirals").select("id,name,category,created_at")
      .eq("user_id", user.user_id).eq("resolved", false).order("created_at", { ascending: false }).limit(1);
    const s = (data ?? [])[0];
    return json({ spiral: s ? { id: s.id, name: s.name ?? "your spiral", category: s.category, created_at: s.created_at } : null, is_pro: isPro(user.plan_tier) });
  },

  // ---- thinkpass ----
  "thinkpass.tiers": async ({ user }) => {
    const level = levelFromXp(user.xp ?? 0);
    let unlocked = user.unlocked_items ?? []; if (typeof unlocked === "string") { try { unlocked = JSON.parse(unlocked); } catch { unlocked = []; } }
    const tiers = LEVEL_UNLOCKS.map(([lv, kind, value, label]) => ({
      level: lv, kind, value, label,
      claimed: unlocked.includes(`${kind}:${value}`),
      ready: level >= lv, locked: level < lv,
    }));
    return json({ tiers, current_level: level });
  },
  "thinkpass.claim": async ({ user, body }) => {
    const level = levelFromXp(user.xp ?? 0);
    const tier = LEVEL_UNLOCKS.find(([lv]) => lv === body.level);
    if (!tier) return json({ detail: "Tier not found" }, 404);
    if (level < tier[0]) return json({ detail: "Tier not reached" }, 400);
    let unlocked = user.unlocked_items ?? []; if (typeof unlocked === "string") { try { unlocked = JSON.parse(unlocked); } catch { unlocked = []; } }
    const item = `${tier[1]}:${tier[2]}`;
    if (!unlocked.includes(item)) unlocked.push(item);
    await db.from("users").update({ unlocked_items: unlocked }).eq("user_id", user.user_id);
    return json({ ok: true, claimed_tier: tier[0], item, unlocked_items: unlocked });
  },

  // ---- phone OTP (dev — no SMS provider wired) ----
  "phone.send": async () => json({ ok: true, message: "OTP sending isn't configured.", dev_otp: "000000" }),
  "phone.verify": async ({ user }) => json({ user: userPublic(user) }),

  // ---- dev tools ----
  "dev.grant_pro": async ({ user }) => {
    await db.from("users").update({ plan_tier: "pro_monthly", has_ever_subscribed: true }).eq("user_id", user.user_id);
    const { data: u } = await db.from("users").select("*").eq("user_id", user.user_id).single();
    return json({ user: userPublic(u) });
  },
  "dev.revoke_pro": async ({ user }) => {
    await db.from("users").update({ plan_tier: "free" }).eq("user_id", user.user_id);
    const { data: u } = await db.from("users").select("*").eq("user_id", user.user_id).single();
    return json({ user: userPublic(u) });
  },
  "dev.reset_xp": async ({ user }) => {
    await db.from("users").update({ xp: 0, level: 1, unlocked_items: [] }).eq("user_id", user.user_id);
    const { data: u } = await db.from("users").select("*").eq("user_id", user.user_id).single();
    return json({ user: userPublic(u) });
  },

  // ---- payments (RevenueCat) ----
  // Purchases happen natively in the app via the RevenueCat SDK; Pro is
  // granted server-side by the separate "revenuecat" webhook function.
  // packages stays here so the paywall can render plan cards.
  "payments.packages": async () => json({
    packages: [
      { id: "weekly", label: "Weekly", amount: 1.99, currency: "usd", tier: "pro_weekly" },
      { id: "monthly", label: "Monthly", amount: 4.99, currency: "usd", tier: "pro_monthly" },
      { id: "lifetime", label: "Lifetime", amount: 49.99, currency: "usd", tier: "lifetime" },
    ],
  }),
  "payments.checkout": async () => json({ detail: "Upgrades now happen through in-app purchase. Tap a plan to buy.", revenuecat: true }, 400),
  "payments.status": async ({ user }) => json({ plan_tier: user.plan_tier ?? "free", is_pro: isPro(user.plan_tier) }),
  "payments.portal": async () => json({ detail: "Manage your subscription in the App Store / Play Store settings.", revenuecat: true }, 400),

  // ---- me ----
  "me": async ({ user }) => json({ user: userPublic(user) }),

  "preferences": async ({ user, body }) => {
    const patch: any = {};
    for (const k of ["tone_preference", "default_category", "bio", "personality", "spiral_frequency"]) {
      if (k in body) patch[k] = body[k];
    }
    if (Object.keys(patch).length) await db.from("users").update(patch).eq("user_id", user.user_id);
    const { data: u } = await db.from("users").select("*").eq("user_id", user.user_id).single();
    return json({ user: userPublic(u) });
  },

  "customize": async ({ user, body }) => {
    if (!isPro(user.plan_tier)) return json({ detail: "Pro plan required for customization" }, 402);
    const cur = (user.customization && typeof user.customization === "object") ? { ...user.customization } : {};
    for (const k of ["active_title", "name_color", "card_theme"]) {
      if (k in body) { if (body[k] === null) delete cur[k]; else cur[k] = body[k]; }
    }
    await db.from("users").update({ customization: cur }).eq("user_id", user.user_id);
    const { data: u } = await db.from("users").select("*").eq("user_id", user.user_id).single();
    return json({ user: userPublic(u) });
  },

  // ---- spirals ----
  "spirals.list": async ({ user, body }) => {
    let q = db.from("spirals").select("*").eq("user_id", user.user_id).order("created_at", { ascending: false }).limit(body.limit ?? 500);
    if (body.category) q = q.eq("category", body.category);
    if (body.resolved === true) q = q.eq("resolved", true);
    if (body.resolved === false) q = q.eq("resolved", false);
    const { data } = await q;
    return json({ spirals: data ?? [] });
  },
  "spirals.get": async ({ user, body }) => {
    const { data } = await db.from("spirals").select("*").eq("id", body.id).eq("user_id", user.user_id).maybeSingle();
    if (!data) return json({ detail: "Spiral not found" }, 404);
    return json({ spiral: data });
  },
  "spirals.resolve": async ({ user, body }) => {
    const resolved = !!body.resolved;
    const { error } = await db.from("spirals").update({
      resolved, resolution_status: body.resolution_status ?? null,
      resolution_note: body.resolution_note ?? null,
      resolved_at: (resolved || body.resolution_status) ? nowIso() : null,
    }).eq("id", body.id).eq("user_id", user.user_id);
    if (error) return json({ detail: String(error.message) }, 500);
    const { data } = await db.from("spirals").select("*").eq("id", body.id).maybeSingle();
    if (!data) return json({ detail: "Spiral not found" }, 404);
    return json({ spiral: data });
  },
  "spirals.patch": async ({ user, body }) => {
    const p = body.patch ?? {};
    const allowed: any = {};
    for (const k of ["category", "tags", "flagged", "folder_id", "accent_color", "name"]) if (k in p) allowed[k] = p[k];
    await db.from("spirals").update(allowed).eq("id", body.id).eq("user_id", user.user_id);
    const { data } = await db.from("spirals").select("*").eq("id", body.id).maybeSingle();
    if (!data) return json({ detail: "Spiral not found" }, 404);
    return json({ spiral: data });
  },
  "spirals.delete": async ({ user, body }) => {
    await db.from("spirals").delete().eq("id", body.id).eq("user_id", user.user_id);
    await db.from("item_pairs").delete().eq("user_id", user.user_id)
      .or(`and(a_type.eq.spiral,a_id.eq.${body.id}),and(b_type.eq.spiral,b_id.eq.${body.id})`);
    return json({ ok: true });
  },
  "spirals.share": async ({ user, body }) => {
    const { data } = await db.from("spirals").select("share_count").eq("id", body.id).eq("user_id", user.user_id).maybeSingle();
    if (!data) return json({ detail: "Spiral not found" }, 404);
    await db.from("spirals").update({ share_count: (data.share_count ?? 0) + 1 }).eq("id", body.id);
    return json({ ok: true });
  },
  "spirals.stats": async ({ user }) => {
    const { data } = await db.from("spirals").select("resolved").eq("user_id", user.user_id);
    const total = data?.length ?? 0;
    const resolved = (data ?? []).filter((s: any) => s.resolved).length;
    return json({ total, resolved, resolved_pct: total ? Math.round((resolved / total) * 100) : 0 });
  },

  // ---- folders ----
  "folders.list": async ({ user }) => {
    const { data: folders } = await db.from("folders").select("*").eq("user_id", user.user_id).order("created_at", { ascending: false });
    const out = [];
    for (const f of folders ?? []) {
      const [sp, tc, cp] = await Promise.all([
        db.from("spirals").select("id", { count: "exact", head: true }).eq("folder_id", f.id).eq("user_id", user.user_id),
        db.from("text_checks").select("id", { count: "exact", head: true }).eq("folder_id", f.id).eq("user_id", user.user_id),
        db.from("compatibility_tests").select("id", { count: "exact", head: true }).eq("folder_id", f.id).eq("user_id", user.user_id),
      ]);
      const spiral_count = sp.count ?? 0, text_check_count = tc.count ?? 0, compat_count = cp.count ?? 0;
      out.push({ ...f, spiral_count, text_check_count, compat_count, item_count: spiral_count + text_check_count + compat_count });
    }
    return json({ folders: out });
  },
  "folders.create": async ({ user, body }) => {
    const name = String(body.name ?? "").trim();
    if (!name) return json({ detail: "Folder name required." }, 400);
    const id = rid("fd");
    await db.from("folders").insert({ id, user_id: user.user_id, name: name.slice(0, 80), color: body.color ?? null, created_at: nowIso() });
    const { data } = await db.from("folders").select("*").eq("id", id).single();
    return json({ folder: data });
  },
  "folders.patch": async ({ user, body }) => {
    const p = body.patch ?? {};
    const upd: any = {};
    if (p.name != null) upd.name = String(p.name).trim().slice(0, 80);
    if ("color" in p) upd.color = p.color;
    if (Object.keys(upd).length) await db.from("folders").update(upd).eq("id", body.id).eq("user_id", user.user_id);
    const { data } = await db.from("folders").select("*").eq("id", body.id).maybeSingle();
    if (!data) return json({ detail: "Folder not found" }, 404);
    return json({ folder: data });
  },
  "folders.delete": async ({ user, body }) => {
    await db.from("spirals").update({ folder_id: null }).eq("folder_id", body.id).eq("user_id", user.user_id);
    await db.from("text_checks").update({ folder_id: null }).eq("folder_id", body.id).eq("user_id", user.user_id);
    await db.from("compatibility_tests").update({ folder_id: null }).eq("folder_id", body.id).eq("user_id", user.user_id);
    await db.from("folders").delete().eq("id", body.id).eq("user_id", user.user_id);
    return json({ ok: true });
  },

  // ---- text-checks (the AI-creating endpoint lives in the "ai" function) ----
  "textchecks.list": async ({ user, body }) => {
    let q = db.from("text_checks").select("*").eq("user_id", user.user_id).order("created_at", { ascending: false }).limit(body.limit ?? 200);
    if (body.folder_id) q = q.eq("folder_id", body.folder_id);
    if (body.flagged === true) q = q.eq("flagged", true);
    const { data } = await q;
    return json({ text_checks: data ?? [] });
  },
  "textchecks.get": async ({ user, body }) => {
    const { data } = await db.from("text_checks").select("*").eq("id", body.id).eq("user_id", user.user_id).maybeSingle();
    if (!data) return json({ detail: "Text-check not found" }, 404);
    return json({ text_check: data });
  },
  "textchecks.patch": async ({ user, body }) => {
    const p = body.patch ?? {};
    const upd: any = {};
    for (const k of ["name", "folder_id", "accent_color", "flagged"]) if (k in p) upd[k] = p[k];
    await db.from("text_checks").update(upd).eq("id", body.id).eq("user_id", user.user_id);
    const { data } = await db.from("text_checks").select("*").eq("id", body.id).maybeSingle();
    if (!data) return json({ detail: "Text-check not found" }, 404);
    return json({ text_check: data });
  },
  "textchecks.resolve": async ({ user, body }) => {
    const resolved = !!body.resolved;
    await db.from("text_checks").update({
      resolved, resolution_status: body.resolution_status ?? null,
      resolution_note: body.resolution_note ?? null,
      resolved_at: (resolved || body.resolution_status) ? nowIso() : null,
    }).eq("id", body.id).eq("user_id", user.user_id);
    const { data } = await db.from("text_checks").select("*").eq("id", body.id).maybeSingle();
    if (!data) return json({ detail: "Text-check not found" }, 404);
    return json({ text_check: data });
  },
  "textchecks.delete": async ({ user, body }) => {
    await db.from("text_checks").delete().eq("id", body.id).eq("user_id", user.user_id);
    await db.from("item_pairs").delete().eq("user_id", user.user_id)
      .or(`and(a_type.eq.text_check,a_id.eq.${body.id}),and(b_type.eq.text_check,b_id.eq.${body.id})`);
    return json({ ok: true });
  },
  "textchecks.save": async ({ user, body }) => {
    if (!isPro(user.plan_tier)) return json({ detail: "Pro required" }, 403);
    const id = rid("tc");
    const { error } = await db.from("text_checks").insert({
      id, user_id: user.user_id, name: body.name ?? null, draft: body.draft,
      context: body.context ?? "", relationship: body.relationship ?? "someone",
      result: body.result, created_at: nowIso(),
    });
    if (error) return json({ detail: `Save failed: ${error.message}` }, 500);
    return json({ text_check_id: id });
  },

  // ---- compatibilities ----
  "compat.list": async ({ user, body }) => {
    let q = db.from("compatibility_tests").select("*").eq("user_id", user.user_id).order("created_at", { ascending: false }).limit(body.limit ?? 200);
    if (body.folder_id) q = q.eq("folder_id", body.folder_id);
    if (body.flagged === true) q = q.eq("flagged", true);
    const { data } = await q;
    return json({ compatibilities: data ?? [] });
  },
  "compat.get": async ({ user, body }) => {
    const { data } = await db.from("compatibility_tests").select("*").eq("id", body.id).eq("user_id", user.user_id).maybeSingle();
    if (!data) return json({ detail: "Compatibility test not found" }, 404);
    return json({ compatibility: data });
  },
  "compat.patch": async ({ user, body }) => {
    const p = body.patch ?? {};
    const upd: any = {};
    for (const k of ["name", "folder_id", "accent_color", "flagged"]) if (k in p) upd[k] = p[k];
    await db.from("compatibility_tests").update(upd).eq("id", body.id).eq("user_id", user.user_id);
    const { data } = await db.from("compatibility_tests").select("*").eq("id", body.id).maybeSingle();
    if (!data) return json({ detail: "Compatibility test not found" }, 404);
    return json({ compatibility: data });
  },
  "compat.resolve": async ({ user, body }) => {
    const resolved = !!body.resolved;
    await db.from("compatibility_tests").update({
      resolved, resolution_status: body.resolution_status ?? null,
      resolution_note: body.resolution_note ?? null,
      resolved_at: (resolved || body.resolution_status) ? nowIso() : null,
    }).eq("id", body.id).eq("user_id", user.user_id);
    const { data } = await db.from("compatibility_tests").select("*").eq("id", body.id).maybeSingle();
    if (!data) return json({ detail: "Compatibility test not found" }, 404);
    return json({ compatibility: data });
  },
  "compat.delete": async ({ user, body }) => {
    await db.from("compatibility_tests").delete().eq("id", body.id).eq("user_id", user.user_id);
    await db.from("item_pairs").delete().eq("user_id", user.user_id)
      .or(`and(a_type.eq.compatibility,a_id.eq.${body.id}),and(b_type.eq.compatibility,b_id.eq.${body.id})`);
    return json({ ok: true });
  },
  "compat.save": async ({ user, body }) => {
    if (!isPro(user.plan_tier)) return json({ detail: "Pro required" }, 403);
    const id = rid("cp");
    const { error } = await db.from("compatibility_tests").insert({
      id, user_id: user.user_id, name: body.name ?? null,
      person_a: body.person_a, person_b: body.person_b, result: body.result, created_at: nowIso(),
    });
    if (error) return json({ detail: `Save failed: ${error.message}` }, 500);
    return json({ compatibility_id: id });
  },

  // ---- pairs ----
  "pairs.create": async ({ user, body }) => {
    const id = rid("pr");
    const { error } = await db.from("item_pairs").insert({
      id, user_id: user.user_id, a_type: body.a_type, a_id: body.a_id,
      b_type: body.b_type, b_id: body.b_id, note: body.note ?? null, created_at: nowIso(),
    });
    if (error) return json({ detail: String(error.message) }, 500);
    const { data } = await db.from("item_pairs").select("*").eq("id", id).single();
    return json({ pair: data, created: true });
  },
  "pairs.delete": async ({ user, body }) => {
    await db.from("item_pairs").delete().eq("id", body.pair_id).eq("user_id", user.user_id);
    return json({ ok: true });
  },
  "pairs.list": async ({ user, body }) => {
    const { data: rows } = await db.from("item_pairs").select("*").eq("user_id", user.user_id)
      .or(`and(a_type.eq.${body.item_type},a_id.eq.${body.item_id}),and(b_type.eq.${body.item_type},b_id.eq.${body.item_id})`);
    const pairs = [];
    for (const r of rows ?? []) {
      const isA = r.a_type === body.item_type && r.a_id === body.item_id;
      const pt = isA ? r.b_type : r.a_type, pid = isA ? r.b_id : r.a_id;
      let preview: any = null;
      if (pt === "spiral") {
        const { data: pr } = await db.from("spirals").select("id,name,situation_text,category,created_at").eq("id", pid).eq("user_id", user.user_id).maybeSingle();
        if (pr) preview = { id: pr.id, name: pr.name, title: (pr.name || pr.situation_text || "").slice(0, 80), subtitle: pr.category, created_at: pr.created_at };
      } else if (pt === "text_check") {
        const { data: pr } = await db.from("text_checks").select("id,name,draft,relationship,created_at").eq("id", pid).eq("user_id", user.user_id).maybeSingle();
        if (pr) preview = { id: pr.id, name: pr.name, title: (pr.name || pr.draft || "").slice(0, 80), subtitle: `Text → ${pr.relationship || "someone"}`, created_at: pr.created_at };
      } else if (pt === "compatibility") {
        const { data: pr } = await db.from("compatibility_tests").select("id,name,person_a,person_b,created_at").eq("id", pid).eq("user_id", user.user_id).maybeSingle();
        if (pr) { const pa = pr.person_a || {}, pb = pr.person_b || {}; preview = { id: pr.id, name: pr.name, title: pr.name || `${pa.name || "?"} × ${pb.name || "?"}`, subtitle: "Compatibility", created_at: pr.created_at }; }
      }
      if (preview) pairs.push({ ...r, partner_type: pt, partner_id: pid, preview });
    }
    return json({ pairs });
  },

  // ---- streak heatmap ----
  "streak.activity": async ({ user, body }) => {
    const days = Math.max(7, Math.min(parseInt(body.days) || 84, 365));
    const start = new Date(); start.setDate(start.getDate() - (days - 1));
    const startStr = start.toISOString().slice(0, 10);
    const { data: rows } = await db.from("streak_activity").select("*").eq("user_id", user.user_id).gte("activity_date", startStr);
    const byDate: Record<string, any> = {};
    for (const r of rows ?? []) byDate[String(r.activity_date).slice(0, 10)] = r;
    const out = [];
    for (let i = 0; i < days; i++) {
      const d = new Date(start); d.setDate(start.getDate() + i);
      const ds = d.toISOString().slice(0, 10);
      const cell = byDate[ds];
      if (!cell) out.push({ date: ds, kind: "miss", total: 0, spiral: 0, text_check: 0, compat: 0 });
      else {
        const sp = cell.spiral_count ?? 0, tc = cell.text_check_count ?? 0, cp = cell.compat_count ?? 0;
        out.push({ date: ds, kind: cell.kind, total: sp + tc + cp, spiral: sp, text_check: tc, compat: cp });
      }
    }
    return json({ days: out, streak_count: user.streak_count ?? 0, freezes_remaining: user.streak_freezes_remaining ?? 0, is_pro: isPro(user.plan_tier) });
  },
};

// ── entry ────────────────────────────────────────────────────────────────
Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  if (req.method !== "POST") return json({ detail: "Method not allowed" }, 405);
  try {
    const body = await req.json();
    const action = String(body.action ?? "");
    const h = handlers[action];
    if (!h) return json({ detail: `Unknown action: ${action}` }, 400);
    // Login actions don't require an existing session; everything else does.
    let user = null;
    if (!UNAUTHED.has(action)) {
      user = await getUser(req);
      if (!user) return json({ detail: "Not authenticated" }, 401);
    }
    return await h({ user, body, req });
  } catch (e) {
    console.error("[api] error:", e);
    return json({ detail: `Server error: ${e}` }, 500);
  }
});
