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
type Ctx = { user: any; body: any };

const handlers: Record<string, (c: Ctx) => Promise<Response>> = {
  // ---- auth ----
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
    const user = await getUser(req);
    if (!user) return json({ detail: "Not authenticated" }, 401);
    const body = await req.json();
    const action = String(body.action ?? "");
    const h = handlers[action];
    if (!h) return json({ detail: `Unknown action: ${action}` }, 400);
    return await h({ user, body });
  } catch (e) {
    console.error("[api] error:", e);
    return json({ detail: `Server error: ${e}` }, 500);
  }
});
