// Supabase Edge Function: "ai"
// ---------------------------------------------------------------------------
// Migration Phase 1 — ports the heaviest server.py logic (text-check +
// compatibility AI generation) off Render and onto Supabase's free
// serverless runtime. Same behaviour as the FastAPI endpoints:
//   - authenticates via the EXISTING custom session token (no auth rewrite)
//   - Pro-gates (both surfaces are Pro-only)
//   - Gemini → HuggingFace(DeepSeek) fallback chain with parse retries
//   - persists the result to text_checks / compatibility_tests
//   - returns { result, saved_id }
//
// Invoke at: POST {SUPABASE_URL}/functions/v1/ai
// Body: { kind: "text_check", draft, context?, relationship? }
//   or  { kind: "compat", person_a, person_b }
// Header: Authorization: Bearer <session_token>   (the app's existing token)
//
// Secrets this function needs (set in Supabase dashboard → Edge Functions):
//   GEMINI_API_KEY, HF_TOKEN
// SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are injected automatically.
// ---------------------------------------------------------------------------

import { createClient } from "jsr:@supabase/supabase-js@2";

const GEMINI_API_KEY = Deno.env.get("GEMINI_API_KEY") ?? "";
const HF_TOKEN = Deno.env.get("HF_TOKEN") ?? "";
const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SERVICE_ROLE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

const GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"];
const HF_MODELS = [
  "deepseek-ai/DeepSeek-V3-0324:novita",
  "deepseek-ai/DeepSeek-V3-0324:together",
  "deepseek-ai/DeepSeek-R1:nebius",
  "meta-llama/Llama-3.3-70B-Instruct:nebius",
  "Qwen/Qwen2.5-72B-Instruct:nebius",
];

const PRO_TIERS = new Set(["pro_weekly", "pro_monthly", "pro_yearly", "lifetime"]);

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-app-key, content-type, accept",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

const db = createClient(SUPABASE_URL, SERVICE_ROLE, {
  auth: { persistSession: false },
});

// ---- JSON salvage (mirrors _parse_json_loose) --------------------------
function stripFences(t: string): string {
  t = t.trim();
  if (t.startsWith("```")) {
    t = t.replace(/^```[a-zA-Z]*\n?/, "").replace(/```$/, "").trim();
  }
  if (!t.startsWith("{")) {
    const a = t.indexOf("{");
    const b = t.lastIndexOf("}");
    if (a !== -1 && b > a) t = t.slice(a, b + 1);
  }
  return t;
}
function parseLoose(text: string): any {
  if (!text || !text.trim()) throw new Error("empty AI response");
  const cands = [stripFences(text), text.trim()];
  const s = stripFences(text);
  const a = s.indexOf("{"), b = s.lastIndexOf("}");
  if (a !== -1 && b > a) cands.push(s.slice(a, b + 1));
  let lastErr: unknown;
  for (const c of cands) {
    try { return JSON.parse(c); } catch (e) { lastErr = e; }
  }
  throw lastErr ?? new Error("unparseable AI response");
}

// ---- AI lanes ----------------------------------------------------------
async function callGemini(prompt: string, maxTokens: number, temp: number): Promise<string | null> {
  if (!GEMINI_API_KEY) return null;
  for (const model of GEMINI_MODELS) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const r = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              contents: [{ parts: [{ text: prompt }] }],
              generationConfig: {
                temperature: temp, topP: 0.95, topK: 40,
                maxOutputTokens: maxTokens, responseMimeType: "application/json",
              },
            }),
          },
        );
        if (!r.ok) { console.log(`[gemini] ${model} HTTP ${r.status}`); continue; }
        const data = await r.json();
        const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
        if (text) { console.log(`[gemini] ✅ ${model}`); return text; }
      } catch (e) {
        console.log(`[gemini] ${model} attempt ${attempt} failed: ${e}`);
      }
    }
  }
  return null;
}

async function callHF(prompt: string, maxTokens: number, temp: number): Promise<string | null> {
  if (!HF_TOKEN) { console.log("[hf] no HF_TOKEN"); return null; }
  const nudge = "\n\nRespond with VALID JSON ONLY. No prose, no markdown fences.";
  for (const model of HF_MODELS) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const r = await fetch("https://router.huggingface.co/v1/chat/completions", {
          method: "POST",
          headers: { Authorization: `Bearer ${HF_TOKEN}`, "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            messages: [{ role: "user", content: prompt + nudge }],
            max_tokens: maxTokens, temperature: temp,
            response_format: { type: "json_object" },
          }),
        });
        if (!r.ok) { console.log(`[hf] ${model} HTTP ${r.status}`); continue; }
        const data = await r.json();
        const text = (data?.choices?.[0]?.message?.content ?? "").trim();
        if (text) { console.log(`[hf] ✅ ${model}`); return text; }
      } catch (e) {
        console.log(`[hf] ${model} attempt ${attempt} failed: ${e}`);
      }
    }
  }
  return null;
}

async function runWithFallback(prompt: string, maxTokens: number, temp: number): Promise<string | null> {
  for (let pass = 0; pass < 3; pass++) {
    const g = await callGemini(prompt, maxTokens, temp);
    if (g) return g;
    const h = await callHF(prompt, maxTokens, temp);
    if (h) return h;
    if (pass < 2) await new Promise((res) => setTimeout(res, 1500 * (pass + 1)));
  }
  return null;
}

// ---- Prompts (faithful to server.py) -----------------------------------
function textCheckPrompt(draft: string, context: string, relationship: string): string {
  const rel = (relationship || "someone").trim();
  const ctx = context.trim() ? `\nContext from the sender: """${context.trim()}"""` : "";
  return `You are reading a draft message someone is about to send to ${rel}. Predict three plausible responses (best/likely/worst) and give a one-line VERDICT: SEND / WAIT / REWRITE.

RULES:
- No platitudes ("trust your gut").
- Don't use "anxious", "spiral", "overthinking".
- Pick a verdict and own it. No hedging.
- Be specific; quote phrases from the draft when helpful.

OUTPUT JSON ONLY:
{
  "predicted_responses": [
    {"severity":"best","title":"3-6 words","reply_text":"1-2 sentence reply in ${rel}'s voice","probability":<int>,"what_it_means":"one sentence"},
    {"severity":"likely","title":"...","reply_text":"...","probability":<int>,"what_it_means":"..."},
    {"severity":"worst","title":"...","reply_text":"...","probability":<int>,"what_it_means":"..."}
  ],
  "verdict":"SEND"|"WAIT"|"REWRITE",
  "verdict_reason":"one sentence, no hedging",
  "rewrite_suggestion":"<string or null, only when REWRITE>",
  "tone_read":"one sentence on how it reads to ${rel}"
}
Probabilities MUST sum to 100. severity values lowercase.

Sending to: ${rel}${ctx}

DRAFT:
"""${draft.trim()}"""

Respond with JSON only.`;
}

function compatBlock(label: string, p: any): string {
  const bits = [`Name: ${(p.name ?? "").trim()}`];
  if (p.gender) bits.push(`Gender: ${p.gender}`);
  if (p.status) bits.push(`Current relationship status: ${String(p.status).trim()}`);
  if (p.description) bits.push(`Description: ${String(p.description).trim()}`);
  if (p.hobbies) bits.push(`Hobbies / interests: ${String(p.hobbies).trim()}`);
  return `PERSON ${label}:\n${bits.join("\n")}`;
}

function compatPrompt(a: any, b: any): string {
  return `You are a sharp, honest, slightly funny observer of human dynamics. Rate how compatible two people are across SEVEN relationship axes. Commit to scores and verdicts — do not hedge. Don't moralise about non-traditional relationships. Don't mention protected attributes.

${compatBlock("A", a)}

${compatBlock("B", b)}

For each axis return score (0-100), verdict (4-8 words), reasoning (1-2 sentences, reference specifics).

OUTPUT JSON ONLY:
{
  "overall_chemistry": <int 0-100>,
  "headline": "one sentence, 10-14 words",
  "axes": [
    {"axis":"friends","score":<int>,"verdict":"...","reasoning":"..."},
    {"axis":"lovers","score":<int>,"verdict":"...","reasoning":"..."},
    {"axis":"friends_with_benefits","score":<int>,"verdict":"...","reasoning":"..."},
    {"axis":"dating","score":<int>,"verdict":"...","reasoning":"..."},
    {"axis":"marriage","score":<int>,"verdict":"...","reasoning":"..."},
    {"axis":"roommates","score":<int>,"verdict":"...","reasoning":"..."},
    {"axis":"work_partners","score":<int>,"verdict":"...","reasoning":"..."}
  ],
  "green_flags": ["...","..."],
  "red_flags": ["...","..."]
}
Respond with JSON only.`;
}

// ---- Fallback payloads -------------------------------------------------
function textCheckFallback(reason: string) {
  return {
    predicted_responses: [
      { severity: "best", title: "Warm reply", reply_text: "They respond positively and the conversation moves on.", probability: 30, what_it_means: "Best case — no harm done." },
      { severity: "likely", title: "Polite acknowledgement", reply_text: "They reply briefly but don't engage with the deeper point.", probability: 50, what_it_means: "Most likely — it lands but doesn't open the conversation you wanted." },
      { severity: "worst", title: "Cool distance", reply_text: "They don't reply, or reply curtly.", probability: 20, what_it_means: "Worst case — the silence becomes its own message." },
    ],
    verdict: "WAIT",
    verdict_reason: "AI is busy — sleep on it and re-check shortly.",
    rewrite_suggestion: null,
    tone_read: "Couldn't read the tone right now.",
    _ai_source: `fallback: ${reason}`,
  };
}
function compatFallback(reason: string) {
  const axes = ["friends", "lovers", "friends_with_benefits", "dating", "marriage", "roommates", "work_partners"];
  return {
    overall_chemistry: 50,
    headline: "AI is busy — read this as a placeholder, not a verdict.",
    axes: axes.map((ax) => ({ axis: ax, score: 50, verdict: "Inconclusive right now.", reasoning: "Couldn't analyse this moment." })),
    green_flags: [], red_flags: [],
    _ai_source: `fallback: ${reason}`,
  };
}

// ---- Spiral prompt (loaded from app_config, base64 JSON of the exact
//      server.py templates so quality is byte-identical) ----------------
let SPIRAL_PROMPTS: Record<string, string> | null = null;
async function getSpiralPrompts(): Promise<Record<string, string>> {
  if (SPIRAL_PROMPTS) return SPIRAL_PROMPTS;
  const { data } = await db.from("app_config").select("value").eq("key", "spiral_prompts").maybeSingle();
  if (!data?.value) throw new Error("spiral prompts not configured");
  const bin = atob(data.value);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  SPIRAL_PROMPTS = JSON.parse(new TextDecoder().decode(bytes));
  return SPIRAL_PROMPTS!;
}
function buildSpiralPrompt(tmpl: string, situation: string, category: string): string {
  return tmpl.split("__SITUATION__").join(situation).split("__CATEGORY__").join(category || "");
}
function spiralFallback(reason: string) {
  return {
    name: "Untitled",
    outcomes: [
      { title: "It lands quieter than your brain says", description: "The most likely version is the boring one: this resolves without the catastrophe your mind has been rehearsing.", probability: 60, severity: "likely", reality_check: "Most worries expire before they ever happen." },
      { title: "It actually goes well", description: "There's a real chance this turns out fine, even good.", probability: 25, severity: "best", reality_check: "You're allowed to plan for the good version too." },
      { title: "The feared version", description: "Even if the worst happens, you'd handle it the way you've handled everything before.", probability: 15, severity: "worst", reality_check: "Knowing the bad version early buys you nothing." },
    ],
    in_control: ["What you do in the next hour", "Whether you keep re-reading this"],
    out_of_control: ["What other people decide", "The timing of someone else's reply"],
    action_steps: ["Put the phone down for ten minutes", "Write the one sentence you actually want to say", "Come back to this tomorrow, not tonight"],
    verdict: { verdict_text: "Your worry will keep. Sleep on it." },
    _ai_source: `fallback: ${reason}`,
  };
}

// ---- Streak + activity bump (port of bump_streak_and_total) ------------
async function recordActivity(uid: string, today: string, source: string, freezeDates: string[]) {
  const col = source === "spiral" ? "spiral_count" : source === "text_check" ? "text_check_count" : "compat_count";
  const { data: ex } = await db.from("streak_activity").select("*").eq("user_id", uid).eq("activity_date", today).maybeSingle();
  if (ex) {
    const upd: any = { kind: ex.kind === "freeze" ? "active" : ex.kind };
    upd[col] = (ex[col] ?? 0) + 1;
    await db.from("streak_activity").update(upd).eq("user_id", uid).eq("activity_date", today);
  } else {
    const ins: any = { user_id: uid, activity_date: today, kind: "active", spiral_count: 0, text_check_count: 0, compat_count: 0 };
    ins[col] = 1;
    await db.from("streak_activity").insert(ins);
  }
  for (const d of freezeDates) {
    const { data: fe } = await db.from("streak_activity").select("user_id").eq("user_id", uid).eq("activity_date", d).maybeSingle();
    if (!fe) await db.from("streak_activity").insert({ user_id: uid, activity_date: d, kind: "freeze", spiral_count: 0, text_check_count: 0, compat_count: 0 });
  }
}
async function bumpSpiralStreak(user: any) {
  const today = new Date().toISOString().slice(0, 10);
  const last = user.last_spiral_date;
  const streak = user.streak_count ?? 0;
  const diff = last ? Math.round((Date.parse(today) - Date.parse(String(last).slice(0, 10))) / 86400000) : null;
  const month = new Date().toISOString().slice(0, 7);
  let freezes = user.streak_freezes_month === month ? (user.streak_freezes_remaining ?? 3) : 3;
  const pro = PRO_TIERS.has(user.plan_tier ?? "free");
  let newStreak: number; const freezeDates: string[] = [];
  if (diff === 0) newStreak = streak || 1;
  else if (diff === 1) newStreak = streak + 1;
  else if (diff !== null && diff >= 2 && diff <= 4 && pro && freezes >= diff - 1) {
    for (let d = 1; d < diff; d++) { const dt = new Date(); dt.setDate(dt.getDate() - d); freezeDates.push(dt.toISOString().slice(0, 10)); }
    freezes -= diff - 1; newStreak = streak + 1;
  } else newStreak = 1;
  const usedToday = user.spirals_used_date === today ? (user.spirals_used_today ?? 0) + 1 : 1;
  await db.from("users").update({
    streak_count: newStreak, last_active: new Date().toISOString(), last_spiral_date: today,
    spirals_used_today: usedToday, spirals_used_date: today,
    spirals_total: (user.spirals_total ?? 0) + 1,
    xp: (user.xp ?? 0) + 10,
    streak_freezes_remaining: freezes, streak_freezes_month: month,
  }).eq("user_id", user.user_id);
  await recordActivity(user.user_id, today, "spiral", freezeDates);
}

async function handleSpiral(user: any, body: any) {
  const tone = String(body.tone ?? "balanced");
  const situation = String(body.situation_text ?? "").trim();
  const category = String(body.category ?? "");
  if (!situation) return json({ detail: "Situation is empty" }, 400);
  if ((user.plan_tier ?? "free") === "free" && (user.spirals_total ?? 0) >= 3) {
    return json({ detail: "Free tier limit reached. Upgrade to keep going." }, 402);
  }
  if (tone.startsWith("drop:") && !PRO_TIERS.has(user.plan_tier ?? "free")) {
    return json({ detail: "Themed drops are a Pro feature. Upgrade to use them." }, 403);
  }
  const spiralId = `sp_${crypto.randomUUID().replace(/-/g, "").slice(0, 18)}`;
  await db.from("spirals").insert({
    id: spiralId, user_id: user.user_id, situation_text: situation,
    category, tone_used: tone, status: "processing", created_at: new Date().toISOString(),
  });

  let payload: any = null;
  try {
    const prompts = await getSpiralPrompts();
    const baseTone = (tone in prompts) ? tone : "balanced";
    const prompt = buildSpiralPrompt(prompts[baseTone], situation, category);
    for (let attempt = 0; attempt < 3 && !payload; attempt++) {
      const text = await runWithFallback(prompt, 4096, 1.15);
      if (!text) break;
      try {
        const data = parseLoose(text);
        const outs = data.outcomes;
        if (!Array.isArray(outs) || outs.length < 1) throw new Error("missing outcomes");
        for (const o of outs) {
          o.probability = Math.max(0, Math.min(100, parseInt(o.probability) || 0));
          let sv = String(o.severity ?? "likely").toLowerCase();
          if (!["best", "likely", "worst"].includes(sv)) sv = "likely";
          o.severity = sv; o.title = o.title ?? "Outcome"; o.description = o.description ?? ""; o.reality_check = o.reality_check ?? "";
        }
        const total = outs.reduce((s: number, o: any) => s + o.probability, 0) || 1;
        for (const o of outs) o.probability = Math.round((o.probability * 100) / total);
        const drift = 100 - outs.reduce((s: number, o: any) => s + o.probability, 0);
        if (drift && outs.length) outs[0].probability += drift;
        data.outcomes = outs;
        data.in_control = data.in_control ?? []; data.out_of_control = data.out_of_control ?? []; data.action_steps = data.action_steps ?? [];
        let v = data.verdict ?? {}; if (typeof v === "string") v = { verdict_text: v };
        v.verdict_text = v.verdict_text ?? "Walk off stage.";
        v.action_steps = v.action_steps ?? data.action_steps; v.in_control = v.in_control ?? data.in_control; v.out_of_control = v.out_of_control ?? data.out_of_control;
        data.verdict = v; data._ai_source = "live";
        payload = data;
      } catch (e) { console.log(`[spiral] parse attempt ${attempt}: ${e}`); }
    }
  } catch (e) { console.log(`[spiral] generation error: ${e}`); }
  if (!payload) payload = spiralFallback("all lanes failed");

  let soundtrack: any = null;
  const st = payload.soundtrack;
  if (PRO_TIERS.has(user.plan_tier ?? "free") && st && typeof st === "object") {
    const t = (st.title ?? "").trim(), l1 = (st.line_1 ?? "").trim(), l2 = (st.line_2 ?? "").trim();
    if (t && l1 && l2) soundtrack = { title: t.slice(0, 60), line_1: l1.slice(0, 120), line_2: l2.slice(0, 120) };
  }
  await db.from("spirals").update({
    outcomes: payload.outcomes, verdict: payload.verdict, status: "complete",
    error_message: payload._ai_source === "live" ? "live" : payload._ai_source,
    name: (payload.name ?? "Untitled").trim().slice(0, 32),
    soundtrack,
  }).eq("id", spiralId);

  try { await bumpSpiralStreak(user); } catch (e) { console.log(`[spiral] streak bump failed: ${e}`); }

  const { data: row } = await db.from("spirals").select("*").eq("id", spiralId).maybeSingle();
  return json({ spiral: row });
}

// ---- Auth: resolve the custom session token to a user ------------------
async function getUser(req: Request): Promise<any | null> {
  const auth = req.headers.get("authorization") ?? "";
  const token = auth.toLowerCase().startsWith("bearer ") ? auth.slice(7).trim() : "";
  if (!token) return null;
  const { data: sess } = await db.from("sessions").select("user_id, expires_at").eq("session_token", token).maybeSingle();
  if (!sess) return null;
  const { data: user } = await db.from("users").select("*").eq("user_id", sess.user_id).maybeSingle();
  return user ?? null;
}

function rid(prefix: string): string {
  return `${prefix}_${crypto.randomUUID().replace(/-/g, "").slice(0, 14)}`;
}

// ---- Handlers ----------------------------------------------------------
async function handleTextCheck(user: any, body: any) {
  const draft = String(body.draft ?? "").trim();
  if (!draft) return json({ detail: "Draft is empty" }, 400);

  let result: any | null = null;
  for (let attempt = 0; attempt < 3 && !result; attempt++) {
    const text = await runWithFallback(textCheckPrompt(draft, body.context ?? "", body.relationship ?? "someone"), 2048, 1.05);
    if (!text) break;
    try {
      const data = parseLoose(text);
      const resp = data.predicted_responses;
      if (!Array.isArray(resp) || resp.length !== 3) throw new Error("bad shape");
      let total = resp.reduce((s: number, r: any) => s + (parseInt(r.probability) || 0), 0) || 1;
      for (const r of resp) r.probability = Math.round(((parseInt(r.probability) || 0) * 100) / total);
      const drift = 100 - resp.reduce((s: number, r: any) => s + r.probability, 0);
      if (drift !== 0) { const lk = resp.find((r: any) => r.severity === "likely") ?? resp[0]; lk.probability += drift; }
      let v = String(data.verdict ?? "WAIT").toUpperCase();
      if (!["SEND", "WAIT", "REWRITE"].includes(v)) v = "WAIT";
      data.verdict = v;
      data.predicted_responses = resp;
      data.rewrite_suggestion = v === "REWRITE" ? (data.rewrite_suggestion ?? null) : null;
      data._ai_source = "live";
      result = data;
    } catch (e) { console.log(`[text-check] parse attempt ${attempt} failed: ${e}`); }
  }
  if (!result) result = textCheckFallback("all lanes failed");

  const savedId = rid("tc");
  let persistError: string | null = null;
  try {
    const { error } = await db.from("text_checks").insert({
      id: savedId, user_id: user.user_id, draft,
      context: body.context ?? "", relationship: body.relationship ?? "someone",
      result, created_at: new Date().toISOString(),
    });
    if (error) throw error;
  } catch (e) { persistError = String(e); }

  return json({
    result, saved_id: persistError ? null : savedId, persist_error: persistError,
    usage: { is_pro: true, used_this_month: 0, monthly_limit: null, remaining: null },
  });
}

async function handleCompat(user: any, body: any) {
  const a = body.person_a ?? {}, b = body.person_b ?? {};
  if (!String(a.name ?? "").trim() || !String(b.name ?? "").trim()) {
    return json({ detail: "Both people need a name" }, 400);
  }
  let result: any | null = null;
  for (let attempt = 0; attempt < 3 && !result; attempt++) {
    const text = await runWithFallback(compatPrompt(a, b), 2400, 1.0);
    if (!text) break;
    try {
      const data = parseLoose(text);
      data.overall_chemistry = Math.max(0, Math.min(100, parseInt(data.overall_chemistry) || 50));
      const axes = data.axes;
      if (!Array.isArray(axes) || axes.length === 0) throw new Error("missing axes");
      for (const ax of axes) ax.score = Math.max(0, Math.min(100, parseInt(ax.score) || 50));
      data.axes = axes;
      data.green_flags = data.green_flags ?? [];
      data.red_flags = data.red_flags ?? [];
      data._ai_source = "live";
      result = data;
    } catch (e) { console.log(`[compat] parse attempt ${attempt} failed: ${e}`); }
  }
  if (!result) result = compatFallback("all lanes failed");

  const savedId = rid("cp");
  let persistError: string | null = null;
  try {
    const { error } = await db.from("compatibility_tests").insert({
      id: savedId, user_id: user.user_id, person_a: a, person_b: b,
      result, created_at: new Date().toISOString(),
    });
    if (error) throw error;
  } catch (e) { persistError = String(e); }

  return json({ result, saved_id: persistError ? null : savedId, persist_error: persistError });
}

// ---- Entry -------------------------------------------------------------
Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  if (req.method !== "POST") return json({ detail: "Method not allowed" }, 405);
  try {
    const user = await getUser(req);
    if (!user) return json({ detail: "Not authenticated" }, 401);
    const body = await req.json();
    // Spirals are available to everyone (free tier capped at 3 inside
    // handleSpiral). Text-check + compatibility are Pro-only.
    if (body.kind === "spiral") return await handleSpiral(user, body);
    if (body.kind === "text_check" || body.kind === "compat") {
      if (user.is_guest) return json({ detail: "Sign in to use this feature" }, 403);
      if (!PRO_TIERS.has(user.plan_tier ?? "free")) {
        return json({ detail: "This is a Pro feature. Upgrade for unlimited reads." }, 403);
      }
      return body.kind === "text_check" ? await handleTextCheck(user, body) : await handleCompat(user, body);
    }
    return json({ detail: "Unknown kind" }, 400);
  } catch (e) {
    console.error("[ai] unexpected:", e);
    return json({ detail: `Server error: ${e}` }, 500);
  }
});
