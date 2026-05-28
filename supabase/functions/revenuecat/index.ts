// Supabase Edge Function: "revenuecat"
// ---------------------------------------------------------------------------
// RevenueCat webhook. RevenueCat's servers POST here on every purchase /
// renewal / cancellation / expiration. We map the event to the user's
// plan_tier in the users table.
//
// SETUP (RevenueCat dashboard → Project → Integrations → Webhooks):
//   URL:    https://iripgxaqbhaxsfqsxxso.supabase.co/functions/v1/revenuecat
//   Header: Authorization = <a secret you choose>
// Then set that same secret here:
//   supabase secrets set REVENUECAT_WEBHOOK_TOKEN=<the same secret>
//
// The app MUST call Purchases.logIn(user_id) so RevenueCat's
// `app_user_id` matches our users.user_id.
// ---------------------------------------------------------------------------

import { createClient } from "jsr:@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SERVICE_ROLE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const WEBHOOK_TOKEN = Deno.env.get("REVENUECAT_WEBHOOK_TOKEN") ?? "";
const db = createClient(SUPABASE_URL, SERVICE_ROLE, { auth: { persistSession: false } });

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

// Map a RevenueCat product_id to our plan_tier. Match on substrings so
// you can name your store products freely (e.g. "overthink_pro_monthly").
function tierForProduct(productId: string): string {
  const p = (productId || "").toLowerCase();
  if (p.includes("life")) return "lifetime";
  if (p.includes("year") || p.includes("annual")) return "pro_yearly";
  if (p.includes("week")) return "pro_weekly";
  if (p.includes("month")) return "pro_monthly";
  return "pro_monthly"; // sensible default for an active entitlement
}

// Event types that GRANT or KEEP access. CANCELLATION keeps access until
// the period actually ends (RevenueCat sends EXPIRATION at that point).
const ACTIVE_EVENTS = new Set([
  "INITIAL_PURCHASE", "RENEWAL", "PRODUCT_CHANGE", "UNCANCELLATION",
  "NON_RENEWING_PURCHASE", "SUBSCRIPTION_EXTENDED", "CANCELLATION",
]);
const REVOKE_EVENTS = new Set(["EXPIRATION", "BILLING_ISSUE_DELAYED", "SUBSCRIPTION_PAUSED"]);

Deno.serve(async (req) => {
  if (req.method !== "POST") return json({ detail: "Method not allowed" }, 405);

  // Verify the shared-secret header RevenueCat sends.
  if (WEBHOOK_TOKEN) {
    const auth = req.headers.get("authorization") ?? "";
    if (auth !== WEBHOOK_TOKEN && auth !== `Bearer ${WEBHOOK_TOKEN}`) {
      return json({ detail: "Unauthorized" }, 401);
    }
  }

  let payload: any;
  try { payload = await req.json(); } catch { return json({ detail: "Bad JSON" }, 400); }

  const event = payload?.event ?? payload;
  const type = String(event?.type ?? "").toUpperCase();
  const userId = event?.app_user_id;
  if (!userId) return json({ detail: "No app_user_id" }, 200); // ack so RC doesn't retry forever

  try {
    if (ACTIVE_EVENTS.has(type)) {
      const tier = tierForProduct(event?.product_id ?? "");
      const expMs = event?.expiration_at_ms ?? null;
      const expiresAt = expMs ? new Date(Number(expMs)).toISOString() : null;
      await db.from("users").update({
        plan_tier: tier,
        plan_expires_at: expiresAt,
        has_ever_subscribed: true,
      }).eq("user_id", userId);
      console.log(`[revenuecat] ${type} → ${userId} = ${tier}`);
    } else if (REVOKE_EVENTS.has(type)) {
      await db.from("users").update({ plan_tier: "free", plan_expires_at: null }).eq("user_id", userId);
      console.log(`[revenuecat] ${type} → ${userId} = free`);
    } else {
      console.log(`[revenuecat] ignored event type: ${type}`);
    }
  } catch (e) {
    console.error("[revenuecat] update failed:", e);
    return json({ detail: `update failed: ${e}` }, 500);
  }
  return json({ ok: true });
});
