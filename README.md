# Referral Inbox Triage Agent

A take-home implementation: triage 8 inbox items at a pediatric SLP / OT / PT practice into structured outputs — classification, intake extraction, routing decisions, and draft replies — without auto-sending anything.

## How to Run

```sh
npm install
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
npm run triage
npm run validate
```

`npm run triage` reads `data/inbox.json` and writes `output.json` plus a per-item tool-call trace at `.trace/tool-calls.jsonl`. `npm run validate` checks the output against `schema/output.schema.json` and the batch-level invariants in `src/validate.ts`. `npm test` runs the grounding unit tests (`src/grounding.test.ts`), which exercise the catch paths — including a member ID that is one character off from the source and a fabricated surname — deterministically, where the extracted value and the source can both be controlled.

## Stack and Runtime

TypeScript, Node LTS, [`@anthropic-ai/sdk`](https://github.com/anthropics/anthropic-sdk-typescript), [`zod`](https://github.com/colinhacks/zod).

Model: `claude-sonnet-4-6` (env-overridable via `ANTHROPIC_MODEL`). Runtime: ~2-3 minutes for 8 items.

## Architecture

A hybrid pipeline runs per inbox item, parallelized across the batch — four stages plus a deterministic grounding gate between extract and route:

**1. LLM classify + extract** (`classifyAndExtract`). Anthropic SDK structured output via a Zod schema. Extracts intake fields (child name, DOB or age, parent contact, discipline, diagnosis/concern, payer, member ID), the classification enum, urgency P0–P3, the safeguarding flag, and a `missing_info` list. System prompt has a `cache_control` breakpoint configured; would activate once the prompt exceeds the 2048-token minimum cacheable prefix.

**1.5. Grounding gate** (`verifyGrounding`, in `src/agent.ts`). Between extract and route, a deterministic check confirms each high-cost identity field — `child_name`, `dob_or_age`, `member_id`, `parent_contact` — is traceable to the source text the model read (sender + subject + body). These are the fields the router then acts on with real tools (`verify_insurance(member_id)`, `hold_slot(patient_ref)`, `search_patient`, `draft_message(contact)`), so a fabricated or corrupted value here is exactly the kind of error that lands silently in a patient record and never surfaces as an exception. No second LLM call — re-asking the model that just hallucinated rarely changes its answer; deterministic checks are reproducible, free, and themselves auditable. If a field fails, the agent retries extraction once with a verbatim-copy instruction naming the failed fields; if it still fails, it fails closed (see Guardrails).

**2. Deterministic router** (`routeItem`). Plain TypeScript that branches on classification + tool results. The insurance chain branches on the `verify_insurance` result, not on what the referral document claims: `in_network` → `find_slots` → `hold_slot` on the earliest → intake task; `out_of_network` → policy lookup + billing task only (no hold); `expired` or `unknown` → billing task with the discrepancy surfaced in the rationale. Safeguarding overrides all other classification — escalate P0, lookup the safeguarding policy, neutral acknowledgement only.

**3. LLM draft** (`draftReply`). Anthropic SDK plain text completion, capped at 300 output tokens. Language is detected from `item.body` up front and passed as an authoritative signal to the prompt — the model is explicitly told not to re-decide based on the sender's name or topic. Channel (`portal` / `email` / `phone`) and recipient are likewise derived deterministically and passed to `draft_message`.

**4. Assembler** (`assembleItemOutput`). Pure function. Builds a schema-valid `ItemOutput`, pulling `tools_called` from `getToolCallsForItem(item.id)` unchanged so the audit log matches the trace exactly.

**Why hybrid over a pure LLM agent.** Deterministic routing gives auditable, predictable tool calls — given a classification and a verified insurance status, the same tools fire every time, in the same order. The LLM handles what it is good at: reading messy unstructured text, catching safeguarding disclosures phrased casually, mirroring the sender's language. Code handles what needs reliability: which tools fire, schema validity, PHI discipline.

## Guardrails and Failure Modes

- **Safeguarding.** Any hint of harm, abuse, or unsafe caregiving — even mentioned casually in passing — sets `safeguarding: true`, fires `escalate` with severity P0, loads the safeguarding policy, and constrains the draft to a neutral acknowledgement only. The draft never references the disclosure, the alleged person, or anything else that could tip off an unsafe caregiver who might also see the family's messages.
- **No clinical advice.** Items classified as `clinical_question` route to a clinical-lead screening task. The draft is instructed not to name conditions, not to give developmental milestones, and not to say "probably normal" or "you should be worried" — it offers an evaluation as the path to a real answer.
- **Never confirm appointments.** Slot holds are described as "held pending staff review." The draft prompt explicitly forbids "confirmed," "scheduled," and similar phrasing. Staff finalizes scheduling, not the agent.
- **Language mirroring.** A deterministic heuristic detects language from `item.body` (not the sender name, not the subject), and the result is passed to the draft prompt as an authoritative "Reply language" line. The LLM is told it must not re-decide. This was a real bug caught in iteration: a safeguarding case from an English-writing parent with a Spanish name produced a Spanish reply until the language signal was made authoritative.
- **Grounded extraction (fail-closed).** Before the router acts on a high-cost identity field (`member_id`, `parent_contact`, `child_name`, `dob_or_age`), a deterministic check confirms the value traces to the source — so a hallucinated ID or contact can't silently enter a record. Each field has a format-tolerant check (member IDs ignore dashes, phones compare digits, names match per token, dates match by numeric component or verbatim phrase); ages must appear in an age context so a stray number elsewhere can't satisfy them. On failure the agent retries extraction once with a verbatim-copy instruction; if it still fails it fails closed — no tools fire, the field is listed in `missing_info`, and the reason is recorded in `decision_rationale`. The asymmetry is deliberate: a false alarm costs a few minutes of staff review, but acting on a fabricated value corrupts a patient record. **Safeguarding is the exception** — it always escalates even when a field is ungrounded, since that path uses only the item ID (not the identity fields) and withholding a P0 alert over an extraction problem would be the worse error. Field checks and edge cases are unit-tested in `src/grounding.test.ts`.
- **Human in the loop.** Every item ships with `requires_human_review = true`. `draft_message` records a draft for staff review; it does not send. The pipeline is an assistant, not an autonomous actor.
- **System of record.** `verify_insurance` is the source of truth — when its result conflicts with what the referral document says (expired coverage, out-of-network despite the referrer marking it in-network), the verified result wins and the discrepancy is surfaced in `decision_rationale`. The relevant policy snippet (`insurance`) is loaded so staff has the rule in hand.
- **Calibrated escalation.** P0 is reserved for safeguarding. P1 is same-day operational impact (today's appointment confusion, same-day cancellation, an active complaint needing today's response). P2 is standard intake. P3 is informational. Over-escalation is itself a failure mode in the assignment rubric, so the urgency rubric is conservative outside the safeguarding case.

## What I Chose Not to Build

- **LangChain / LangGraph agent loop.** A graph-based agent loop would generalize routing to inbox shapes the prompt does not anticipate, but adds framework complexity and non-deterministic tool selection. Under a 2-hour time box for a regulated-domain workflow, deterministic TypeScript routing is more auditable and easier to defend in review.
- **PHI encryption at rest.** Synthetic data only per the assignment constraints. In production: field-level encryption before any LLM call, a tightly scoped retention policy on the trace log, and a HIPAA BAA with the LLM provider — none of which is exercised by a synthetic-data take-home.
- **Retry logic on LLM failures.** Currently a thrown error fails the single item (it does not crash the batch — items run via `Promise.all`, so a failure surfaces as a rejected promise for that item alone). Production-grade: exponential backoff with jitter on `classifyAndExtract` and `draftReply` calls, on top of the SDK's built-in 429 / 5xx retries.
- **Confidence scoring on classification.** Low-confidence items could be flagged for human classification review rather than fed into the deterministic router. Currently the router trusts the classifier's output.
- **Grounding beyond field provenance.** The grounding gate verifies that an extracted value came from the source text — not that it is *correct for this patient*. Three boundaries are deliberately out of scope: (1) **Association** — if a message mentions a sibling, a `member_id` can be perfectly grounded yet belong to the wrong child; grounding confirms the value is in the document, not that it was matched to the right person. (2) **Source correctness** — if the referring provider wrote the wrong ID on the fax, we faithfully ground a wrong value. (3) **Word→digit normalization** — a word-form date passed through verbatim ("the third of March, two thousand twenty") grounds correctly, but if the model *normalizes* a word source into a different digit form (`"seven"` → `"7"`) the forms no longer match and it flags as ungrounded (a false positive the verbatim retry usually rescues). A day/month transposition (`09/03` vs `03/09`) is intentionally *not* flagged, since both readings are equally supported by an ambiguous source. The next step here is entity-level association checks, not just field-level provenance.
- **OCR for fax/PDF referrals.** The agent works on text; `InboxItem.body` is assumed to already be OCR'd upstream. A production pipeline would own the scan-to-text step and carry an OCR-confidence signal into the grounding gate.

## What I Would Do With Another 4 Hours

- Lift routing into a **LangGraph** agent so it generalizes to unseen item types without code changes — the deterministic-routing tradeoff above is the right call now, but the inbox will eventually grow shapes the switch statement does not cover.
- Add **LangSmith** (or equivalent) tracing for per-item LLM observability — token usage, latency, prompt-cache hit rate — alongside the existing tool trace.
- Replace the mock tools with **real integrations**: EHR patient lookup, payer eligibility API, calendar system. The current shape (one `withItemContext`-scoped trace per item, schema-validated outputs) is intentionally designed to swap these in without touching the routing.
- Add a **classification confidence threshold**: items below threshold skip routing and go straight to human review with no tool calls fired.
- **Spanish-language classification prompt variant.** When the language detector flags Spanish before the LLM call, swap to a system prompt written natively in Spanish rather than asking an English-trained prompt to extract Spanish-language fields. Catches the long tail of edge cases that bilingual prompting handles imperfectly.
