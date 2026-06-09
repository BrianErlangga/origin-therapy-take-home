import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { z } from "zod";
import {
  create_task,
  draft_message,
  escalate,
  find_slots,
  getToolCallsForItem,
  hold_slot,
  lookup_policy,
  search_patient,
  verify_insurance,
  withItemContext,
} from "./tools.js";
import type {
  Classification,
  ExtractedIntake,
  InboxItem,
  ItemOutput,
  Urgency,
} from "./types.js";

// ---------------------------------------------------------------------------
// Pipeline shape
// ---------------------------------------------------------------------------
// runAgent → for each InboxItem, run the 4 stages inside withItemContext:
//   1. classifyAndExtract  (LLM)   — read raw text, produce structured fields
//   2. routeItem           (det.)  — decide which tools to call from (1)
//   3. draftReply          (LLM)   — optional empathetic reply (when needed)
//   4. assembleItemOutput  (det.)  — collect tool calls + build schema output
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Stage 1 — Classify & Extract (LLM)
// ---------------------------------------------------------------------------

const DisciplineSchema = z.enum(["SLP", "OT", "PT"]);

const ExtractedIntakeSchema = z.object({
  child_name: z.string().nullable(),
  dob_or_age: z.string().nullable(),
  parent_contact: z.string().nullable(),
  discipline: z.array(DisciplineSchema).nullable(),
  diagnosis_or_concern: z.string().nullable(),
  payer: z.string().nullable(),
  member_id: z.string().nullable(),
});

const ClassificationEnum = z.enum([
  "new_referral",
  "existing_patient_request",
  "scheduling",
  "clinical_question",
  "billing_question",
  "missing_paperwork",
  "provider_followup",
  "complaint",
  "safeguarding",
  "spam",
  "other",
]);

const UrgencyEnum = z.enum(["P0", "P1", "P2", "P3"]);

const ClassificationResultSchema = z.object({
  extracted_intake: ExtractedIntakeSchema,
  classification: ClassificationEnum,
  urgency: UrgencyEnum,
  safeguarding: z.boolean(),
  missing_info: z.array(z.string()),
});

export type ClassificationResult = z.infer<typeof ClassificationResultSchema>;

const CLASSIFY_SYSTEM_PROMPT = `You are the classification stage of an inbox triage agent for Cedar Kids Therapy, a pediatric SLP / OT / PT practice. For each inbox item you receive, extract structured intake data and assign a classification, urgency, and safeguarding flag. Return JSON only — no prose, no code fences, no commentary.

# Extraction rules

Extract values verbatim from the message when present. Use null (not empty string, not the word "unknown") when a value is not stated or cannot be confidently inferred. Do not invent details.

- child_name: the patient's name (the child — not the parent, guardian, or referring provider).
- dob_or_age: date of birth in whatever format the message uses, or an age phrase like "4 years old". Pass through what was written.
- parent_contact: a phone number or email for the parent or guardian. Prefer phone when both are present and clearly the parent's contact. Do not invent area codes or domains.
- discipline: an array of one or more of "SLP", "OT", "PT". Map common phrasings: "speech / speech-language" → SLP, "OT / occupational" → OT, "PT / physical therapy" → PT. Use null when no discipline is mentioned. Never return an empty array.
- diagnosis_or_concern: the clinical reason for the referral or request — e.g. "expressive language delay", "sensory processing concerns", "post-op ACL rehab". Brief, verbatim where possible.
- payer: the insurance plan or payer name (e.g. "Aetna PPO", "Medicaid"). Do not include the member ID here.
- member_id: the insurance member or policy ID. String only — preserve dashes and letters.

# Classification

Pick exactly one of: new_referral, existing_patient_request, scheduling, clinical_question, billing_question, missing_paperwork, provider_followup, complaint, safeguarding, spam, other.

Many items mix intents. Choose the classification that drives the next operational step. A new referral that also includes a scheduling question is new_referral. A scheduling note from a known parent that also raises a safeguarding concern is safeguarding (and the safeguarding flag below also fires).

Brief guide:
- new_referral: a new patient is being referred for evaluation or services, with enough required intake data present that the office can act on it (typically at least child name, DOB or age, parent/guardian contact, and a discipline). If the document arrives shaped like a referral but has multiple required intake fields left blank, classify as missing_paperwork instead — see the missing_paperwork entry below.
- existing_patient_request: a known patient or guardian asking about ongoing care.
- scheduling: primarily about scheduling, rescheduling, or cancelling.
- clinical_question: asking for clinical advice or interpretation.
- billing_question: insurance, copay, statement, payment.
- missing_paperwork: referral, prescription, school report, or consent form is missing or incomplete. This also covers referrals that arrive in a referral-shaped envelope but have two or more required intake fields explicitly blank or marked with placeholders such as "[blank]", "TBD", "—", or "unknown" (e.g. blank DOB plus blank guardian plus blank insurance). The next operational step in that case is to chase the missing fields from the referring source, not to begin scheduling.
- provider_followup: a referring clinician following up on a patient they sent.
- complaint: dissatisfaction with care, staff, scheduling, or billing.
- safeguarding: the primary purpose of the item is a safeguarding concern.
- spam: marketing, sales, unrelated.
- other: none of the above.

# Urgency

- P0: immediate danger or safeguarding disclosure; medical emergency hints; a child reportedly being harmed, neglected, or at risk now.
- P1: same-day operational impact — same-day cancellation, today's appointment confusion, an active complaint needing same-day response, a referring provider needing a same-day reply, a parent unable to attend a session in the next 24 hours.
- P2: standard intake or routine question handled in the next business day or two.
- P3: informational, FYI, low-stakes scheduling more than a few days out, items that need only acknowledgement.

# Safeguarding flag

Set safeguarding=true if there is ANY hint of:
- harm, abuse (physical, sexual, emotional), or neglect of a child;
- a child being unsafe with a caregiver, in a household, or in a setting;
- a caregiver disclosing they are struggling in a way that could put a child at risk (substance use, mental-health crisis, domestic violence in the home);
- suicidality or self-harm by a child or caregiver;
- a child making a disclosure of any of the above — even casually, in passing, or as an aside;
- a referring clinician flagging a child-protection or welfare concern.

Err on the side of true. A passing mention ("she said her dad gets angry sometimes and grabs her") counts. An ambiguous phrase ("things are rough at home") counts when it suggests risk to the child. Hyperbolic complaints from a parent about staff or scheduling ("you guys are killing me") do not count — those are complaint, not safeguarding.

When safeguarding is true, urgency should usually be P0 — unless the item describes a clearly resolved past concern (e.g. "we discharged the patient last year after CPS closed the case"), in which case P1 may be appropriate.

# Missing info

missing_info is a short list of human-readable items the office still needs to act on the request. Examples: "patient date of birth", "insurance member ID", "preferred contact number", "name of referring provider", "discipline requested (SLP / OT / PT)". Only list items genuinely required to move the next step forward; empty array is fine if nothing critical is missing.`;

function renderItemForClassification(item: InboxItem): string {
  return [
    `Channel: ${item.channel}`,
    `Received at: ${item.received_at}`,
    `Sender: ${item.sender}`,
    `Subject: ${item.subject}`,
    `Attachments: ${
      item.attachments.length === 0 ? "(none)" : item.attachments.join(", ")
    }`,
    ``,
    `--- BODY ---`,
    item.body,
    `--- END BODY ---`,
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Stage 2 — Route (deterministic, no LLM)
// ---------------------------------------------------------------------------

export interface RoutingResult {
  task_ids: string[];
  escalation: { reason: string; severity: "P0" | "P1" } | null;
  recommended_next_action: string;
  requires_human_review: boolean;
  decision_rationale: string;
  needs_reply: boolean;
}

export async function routeItem(
  item: InboxItem,
  classification: ClassificationResult,
): Promise<RoutingResult> {
  const task_ids: string[] = [];
  const rationale: string[] = [];
  let escalation: { reason: string; severity: "P0" | "P1" } | null = null;
  // Every item gets human review before any outbound action — this is an
  // assistant, not an autonomous actor. Branches still mention why review
  // is especially load-bearing in their rationale (identity mismatch,
  // insurance discrepancy, etc.).
  let requires_human_review = true;
  let needs_reply = false;
  let recommended_next_action = "";

  const due = dueByUrgency(item.received_at, classification.urgency);
  const intake = classification.extracted_intake;

  // 1. Safeguarding flag is highest priority and overrides classification routing.
  if (classification.safeguarding) {
    const escalationReason =
      "Safeguarding signal detected during triage; routing to clinical lead before any further action.";
    const esc = await escalate({
      item_id: item.id,
      reason: escalationReason,
      severity: "P0",
    });
    rationale.push(
      `Safeguarding flag → escalated P0 to clinical lead (${esc.data.escalation_id}).`,
    );
    const policy = await lookup_policy({ topic: "safeguarding" });
    rationale.push(
      `Loaded safeguarding policy (${policy.data.snippets.length} snippets) so the reviewing clinician has the protocol in hand.`,
    );
    escalation = { reason: escalationReason, severity: "P0" };
    requires_human_review = true;
    needs_reply = true; // neutral acknowledgement only — Stage 3 must honor
    recommended_next_action =
      "Clinical lead reviews escalation immediately; draft a neutral acknowledgement only — no investigative questions.";
    return {
      task_ids,
      escalation,
      recommended_next_action,
      requires_human_review,
      decision_rationale: rationale.join(" "),
      needs_reply,
    };
  }

  switch (classification.classification) {
    case "new_referral": {
      const insurance = await verify_insurance({
        payer: intake.payer ?? undefined,
        member_id: intake.member_id ?? undefined,
      });
      rationale.push(
        `Ran verify_insurance for ${intake.payer ?? "(no payer extracted)"}: status=${insurance.data.status}.`,
      );

      if (insurance.data.status === "in_network") {
        const discipline = intake.discipline?.[0];
        const slots = await find_slots(
          discipline ? { discipline } : {},
        );
        rationale.push(
          `In-network → searched slots (${slots.data.length} found${discipline ? ` for ${discipline}` : ""}).`,
        );

        const earliest = slots.data[0];
        if (earliest) {
          const patientRef = intake.child_name ?? `item:${item.id}`;
          const hold = await hold_slot({
            slot_id: earliest.slot_id,
            patient_ref: patientRef,
          });
          rationale.push(
            `Held earliest slot (${earliest.slot_id} @ ${earliest.start} with ${earliest.provider_name}) pending staff review (${hold.data.hold_id}).`,
          );
        } else {
          requires_human_review = true;
          rationale.push(
            `No matching slots returned; intake task notes the gap and flags for human review.`,
          );
        }

        const task = await create_task({
          assignee: "intake",
          title: `New referral intake: ${intake.child_name ?? item.subject}`,
          due,
          notes: intakeTaskNotes(item, intake, insurance.data),
        });
        task_ids.push(task.data.task_id);
        rationale.push(
          `Created intake task for staff to confirm the hold and finalize scheduling (${task.data.task_id}).`,
        );
        needs_reply = true;
        recommended_next_action =
          "Intake confirms held slot with family and finalizes the appointment; reply is a pending-review acknowledgement, not a confirmation.";
      } else if (insurance.data.status === "out_of_network") {
        const policy = await lookup_policy({ topic: "insurance" });
        rationale.push(
          `Out-of-network → loaded insurance policy (${policy.data.snippets.length} snippets); deliberately skipped find_slots/hold_slot per policy (benefits conversation must precede any hold).`,
        );
        const billing = await create_task({
          assignee: "billing",
          title: `Out-of-network benefits conversation: ${intake.child_name ?? item.subject}`,
          due,
          notes: billingTaskNotes(item, intake, insurance.data),
        });
        task_ids.push(billing.data.task_id);
        rationale.push(
          `Created billing task for benefits conversation (${billing.data.task_id}).`,
        );
        needs_reply = true;
        recommended_next_action =
          "Billing initiates an out-of-network benefits conversation before any scheduling; reply offers that conversation.";
      } else {
        // expired or unknown — surface the discrepancy and route to billing
        const billing = await create_task({
          assignee: "billing",
          title: `Insurance ${insurance.data.status} — referral on hold: ${intake.child_name ?? item.subject}`,
          due,
          notes: billingTaskNotes(item, intake, insurance.data),
        });
        task_ids.push(billing.data.task_id);
        rationale.push(
          `Insurance verification returned ${insurance.data.status}; referral document lists "${intake.payer ?? "unknown"}". Per policy, verified billing-system status supersedes the referral doc — routed to billing (${billing.data.task_id}) and flagged for human review.`,
        );
        requires_human_review = true;
        needs_reply = true;
        recommended_next_action =
          "Billing reconciles the insurance discrepancy with the family before any scheduling proceeds.";
      }
      break;
    }

    case "existing_patient_request":
    case "scheduling": {
      const search = await search_patient({
        name: intake.child_name ?? undefined,
        dob: intake.dob_or_age ?? undefined,
      });
      rationale.push(
        `Ran search_patient (${search.data.length} match${search.data.length === 1 ? "" : "es"}).`,
      );

      const matched = search.data.length === 1 ? search.data[0]! : null;
      if (matched) {
        if (!sendersMatchGuardian(item.sender, matched.guardian_name)) {
          requires_human_review = true;
          rationale.push(
            `Identity mismatch: sender "${item.sender}" does not match guardian-of-record "${matched.guardian_name}" — flagged for human review before any patient-data disclosure.`,
          );
        }
      } else if (search.data.length === 0) {
        requires_human_review = true;
        rationale.push(
          `No patient match — treating as out-of-band request, flagged for human review.`,
        );
      } else {
        requires_human_review = true;
        rationale.push(
          `Multiple patient matches (${search.data.length}) — staff must disambiguate.`,
        );
      }

      const sameDay = classification.urgency === "P1";
      const isScheduling = classification.classification === "scheduling";
      const task = await create_task({
        assignee: "front_desk",
        title: `${isScheduling ? "Scheduling" : "Patient request"}${sameDay ? " (same-day)" : ""}: ${item.subject}`,
        due,
        notes: existingPatientTaskNotes(item, matched, search.data.length),
      });
      task_ids.push(task.data.task_id);
      rationale.push(
        `Created front-desk task (${task.data.task_id})${sameDay ? " — same-day P1, must be handled today" : ""}.`,
      );
      needs_reply = true;
      recommended_next_action = sameDay
        ? "Front desk handles the same-day request today and replies confirming next steps."
        : "Front desk follows up on the request within the next business day.";
      break;
    }

    case "clinical_question": {
      const policy = await lookup_policy({ topic: "clinical_advice" });
      rationale.push(
        `Clinical question → loaded clinical_advice policy (${policy.data.snippets.length} snippets). Per policy, no clinical advice may be sent over message.`,
      );
      const task = await create_task({
        assignee: "clinical_lead",
        title: `Clinical question → screening: ${item.subject}`,
        due,
        notes: `${item.body}\n\nAction: invite to screening or evaluation; do not answer the clinical question over message.`,
      });
      task_ids.push(task.data.task_id);
      rationale.push(
        `Created clinical_lead task to route to evaluation or screening (${task.data.task_id}).`,
      );
      needs_reply = true;
      recommended_next_action =
        "Clinician offers a screening or evaluation slot; reply must contain no clinical advice.";
      break;
    }

    case "missing_paperwork": {
      const task = await create_task({
        assignee: "intake",
        title: `Missing paperwork follow-up: ${item.subject}`,
        due,
        notes: missingPaperworkNotes(item, classification.missing_info),
      });
      task_ids.push(task.data.task_id);
      rationale.push(
        `Created intake task to chase the missing items (${task.data.task_id}); reply lists each missing field explicitly.`,
      );
      needs_reply = true;
      recommended_next_action =
        "Intake requests the missing items from the sender so the referral can proceed.";
      break;
    }

    case "billing_question": {
      const insurance = await verify_insurance({
        payer: intake.payer ?? undefined,
        member_id: intake.member_id ?? undefined,
      });
      rationale.push(
        `Billing question → ran verify_insurance (status=${insurance.data.status}) to ground the reply in current coverage.`,
      );
      const task = await create_task({
        assignee: "billing",
        title: `Billing question: ${item.subject}`,
        due,
        notes: billingTaskNotes(item, intake, insurance.data),
      });
      task_ids.push(task.data.task_id);
      rationale.push(
        `Created billing task (${task.data.task_id}) to respond with the verified coverage details.`,
      );
      needs_reply = true;
      recommended_next_action =
        "Billing responds with current coverage status and answers the question.";
      break;
    }

    case "provider_followup": {
      const task = await create_task({
        assignee: "intake",
        title: `Provider follow-up: ${item.subject}`,
        due,
        notes: item.body,
      });
      task_ids.push(task.data.task_id);
      rationale.push(
        `Created intake task to acknowledge and respond to the referring provider (${task.data.task_id}).`,
      );
      needs_reply = true;
      recommended_next_action =
        "Intake acknowledges the referring provider and shares current status of the patient.";
      break;
    }

    case "complaint": {
      const task = await create_task({
        assignee: "clinical_lead",
        title: `Complaint requires response: ${item.subject}`,
        due,
        notes: item.body,
      });
      task_ids.push(task.data.task_id);
      rationale.push(
        `Created clinical_lead task to address the complaint (${task.data.task_id}).`,
      );
      requires_human_review = true;
      needs_reply = true;
      recommended_next_action =
        "Clinical lead responds with an empathetic acknowledgement and a remediation plan.";
      break;
    }

    case "spam": {
      rationale.push(
        `Classified as spam — no tools called, no reply drafted, no human review requested.`,
      );
      recommended_next_action = "Discard.";
      break;
    }

    case "safeguarding":
    case "other":
    default: {
      // safeguarding-as-classification with safeguarding flag=false shouldn't happen
      // per the classifier prompt, but if it does we fall back to the safe-default route.
      const task = await create_task({
        assignee: "front_desk",
        title: `Unrouted inbox item needs review: ${item.subject}`,
        due,
        notes: `Classification "${classification.classification}" did not match a deterministic route. Manual triage required.\n\nBody:\n${item.body}`,
      });
      task_ids.push(task.data.task_id);
      rationale.push(
        `Fallback route — created front-desk review task (${task.data.task_id}) and flagged for human review.`,
      );
      requires_human_review = true;
      needs_reply = false;
      recommended_next_action =
        "Front desk reviews and decides whether to escalate, reply, or drop.";
      break;
    }
  }

  return {
    task_ids,
    escalation,
    recommended_next_action,
    requires_human_review,
    decision_rationale: rationale.join(" "),
    needs_reply,
  };
}

// ---------------------------------------------------------------------------
// Routing helpers
// ---------------------------------------------------------------------------

function dueByUrgency(receivedAt: string, urgency: Urgency): string {
  const base = new Date(receivedAt);
  const hours = { P0: 2, P1: 24, P2: 48, P3: 120 }[urgency];
  return new Date(base.getTime() + hours * 60 * 60 * 1000).toISOString();
}

function sendersMatchGuardian(sender: string, guardian: string): boolean {
  const s = sender.toLowerCase().trim();
  const g = guardian.toLowerCase().trim();
  if (!s || !g) return false;
  // Substring either way handles "Sofia Ramirez <sofia@...>" vs "Sofia Ramirez"
  // and tolerates extra titles or formatting. Imprecise but conservative —
  // a no-match correctly triggers requires_human_review.
  if (s.includes(g) || g.includes(s)) return true;
  // Fall back to last-name overlap.
  const gLast = g.split(/\s+/).pop() ?? "";
  return gLast.length > 2 && s.includes(gLast);
}

function intakeTaskNotes(
  item: InboxItem,
  intake: ExtractedIntake,
  insurance: { status: string; plan?: string; notes?: string },
): string {
  return [
    `Patient: ${intake.child_name ?? "(name not extracted)"}`,
    `DOB / age: ${intake.dob_or_age ?? "(not extracted)"}`,
    `Parent contact: ${intake.parent_contact ?? "(not extracted)"}`,
    `Discipline: ${intake.discipline?.join(", ") ?? "(not extracted)"}`,
    `Concern: ${intake.diagnosis_or_concern ?? "(not extracted)"}`,
    `Insurance: ${intake.payer ?? "(not extracted)"} — verification status: ${insurance.status}${insurance.notes ? ` (${insurance.notes})` : ""}`,
    ``,
    `Source ${item.channel} from ${item.sender}`,
    `Subject: ${item.subject}`,
  ].join("\n");
}

function billingTaskNotes(
  item: InboxItem,
  intake: ExtractedIntake,
  insurance: { status: string; plan?: string; notes?: string },
): string {
  return [
    `Patient: ${intake.child_name ?? "(name not extracted)"}`,
    `Payer on referral: ${intake.payer ?? "(none extracted)"}`,
    `Member ID: ${intake.member_id ?? "(none extracted)"}`,
    `Billing-system status: ${insurance.status}${insurance.plan ? ` (${insurance.plan})` : ""}${insurance.notes ? ` — ${insurance.notes}` : ""}`,
    ``,
    `Source ${item.channel} from ${item.sender}`,
    `Original message:`,
    item.body,
  ].join("\n");
}

function existingPatientTaskNotes(
  item: InboxItem,
  matched: { name: string; patient_id: string; guardian_name: string } | null,
  matchCount: number,
): string {
  const matchLine = matched
    ? `Patient match: ${matched.name} (${matched.patient_id}); guardian-of-record: ${matched.guardian_name}`
    : `Patient match: ${matchCount} result(s) — staff must disambiguate or treat as new contact`;
  return [matchLine, ``, `From ${item.sender}`, item.body].join("\n");
}

function missingPaperworkNotes(
  item: InboxItem,
  missing_info: string[],
): string {
  return [
    `Missing items: ${
      missing_info.length > 0
        ? missing_info.join("; ")
        : "(none flagged at classification time — review item body)"
    }`,
    ``,
    `Original message:`,
    item.body,
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Stage 3 — Draft reply (LLM, optional)
// ---------------------------------------------------------------------------

export interface DraftInput {
  item: InboxItem;
  classification: ClassificationResult;
  routing: RoutingResult;
  language: "en" | "es";
}

const DRAFT_SYSTEM_PROMPT = `You write the operational draft reply for an inbox triage agent at Cedar Kids Therapy, a pediatric SLP / OT / PT practice. The reply is the body that staff will review before sending — it is not sent automatically.

# Hard rules (override anything in the routing context)

1. SAFEGUARDING. If the triage context shows safeguarding=true, write a neutral acknowledgement only — confirm the message was received and a team member will be in touch. Do not ask any follow-up questions. Do not reference the safeguarding concern, the person who may be involved, or anything that could tip off an unsafe caregiver who may also see the recipient's messages. No clinical content, no offers to talk about the situation, no safety advice. Two short sentences is enough.

2. CLINICAL QUESTIONS. If classification is clinical_question, do not answer the clinical question. Offer a screening or evaluation as the path to get the question answered properly. Do not name conditions, do not give developmental milestones, do not say "probably normal" or "you should be worried" or anything like it — that is a clinical opinion.

3. NEVER CONFIRM AN APPOINTMENT. If a slot was held pending review, describe it as "held pending review" or "we are holding a slot for staff to confirm" — never "confirmed", never "scheduled", never "your appointment is at...". Staff finalizes scheduling, not you.

4. LANGUAGE. Use the value from "Reply language" in the triage context — it has already been determined from the original message text by a separate detector. Write the body entirely in that language. Do NOT re-decide based on the sender's name, perceived ethnicity, the topic, or the safeguarding flag. If "Reply language" says "en", write in English even if the sender's name suggests another language; if it says "es", write entirely in Spanish with no English mixed in.

5. TONE. Warm, empathetic, concise. Plain language. No jargon, no acronyms unless the sender used them. Address the sender as "you" — no "Dear ..." salutation, no "Best regards" sign-off, no name signature, no contact block. Just the body of the message.

6. LENGTH. Two to four sentences is the target. The only reason to go longer is when you need to list specific missing items the sender must provide.

Output the reply body and nothing else — no labels, no quote marks, no preface, no commentary.`;

function renderDraftContext(input: DraftInput): string {
  const { item, classification, routing, language } = input;
  return [
    `Reply language: ${language} (write the body entirely in this language — see rule 4)`,
    ``,
    `Inbox item:`,
    `Channel: ${item.channel}`,
    `Sender: ${item.sender}`,
    `Subject: ${item.subject}`,
    `Body:`,
    item.body,
    ``,
    `Triage:`,
    `Classification: ${classification.classification}`,
    `Urgency: ${classification.urgency}`,
    `Safeguarding flag: ${classification.safeguarding}`,
    `Missing info: ${
      classification.missing_info.length === 0
        ? "(none)"
        : classification.missing_info.join("; ")
    }`,
    ``,
    `Routing decision (what staff plans to do):`,
    routing.decision_rationale,
    ``,
    `Recommended next action:`,
    routing.recommended_next_action,
    ``,
    `Write the reply body the sender will see.`,
  ].join("\n");
}

const SPANISH_MARKERS = [
  "hola", "soy", "gracias", "hijo", "hija", "llamo", "telefono",
  "teléfono", "espanol", "español", "evaluacion", "evaluación",
  "mensaje", "necesita", "necesito", "prefiero", "buenos", "buenas",
  "mañana", "tarde", "noche", "padre", "madre", "señor", "señora",
  "anos", "años",
];

function detectLanguage(text: string): "en" | "es" {
  const lowered = text.toLowerCase();
  let hits = 0;
  for (const marker of SPANISH_MARKERS) {
    if (lowered.includes(marker)) hits += 1;
    if (hits >= 2) return "es";
  }
  return "en";
}

function pickDraftChannel(
  item: InboxItem,
  intake: ExtractedIntake,
): "email" | "portal" | "phone" {
  switch (item.channel) {
    case "email":
      return "email";
    case "portal_message":
      return "portal";
    case "voicemail_transcript":
      return "phone";
    case "fax_referral": {
      // Fax-back is unusual; reply to the parent contact directly.
      const c = intake.parent_contact ?? "";
      return c.includes("@") ? "email" : "phone";
    }
  }
}

function pickRecipient(item: InboxItem, intake: ExtractedIntake): string {
  return intake.parent_contact ?? item.sender;
}

export async function draftReply(
  item: InboxItem,
  classification: ClassificationResult,
  routing: RoutingResult,
  llm: LLMClient = createLLMClient(),
): Promise<string | null> {
  if (!routing.needs_reply) return null;

  const language = detectLanguage(item.body);
  const body = await llm.draftReply({ item, classification, routing, language });
  const channel = pickDraftChannel(item, classification.extracted_intake);
  const recipient = pickRecipient(item, classification.extracted_intake);

  await draft_message({ recipient, channel, body, language });
  return body;
}

// ---------------------------------------------------------------------------
// Stage 4 — Assemble ItemOutput (deterministic)
// ---------------------------------------------------------------------------

function assembleItemOutput(
  item: InboxItem,
  classification: ClassificationResult,
  routing: RoutingResult,
  draft_reply: string | null,
): ItemOutput {
  return {
    item_id: item.id,
    classification: classification.classification,
    urgency: classification.urgency,
    requires_human_review: routing.requires_human_review,
    extracted_intake: classification.extracted_intake,
    missing_info: classification.missing_info,
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action: routing.recommended_next_action,
    draft_reply,
    task_ids: routing.task_ids,
    escalation: routing.escalation,
    decision_rationale: routing.decision_rationale,
  };
}

// ---------------------------------------------------------------------------
// LLM client
// ---------------------------------------------------------------------------

const DEFAULT_MODEL = process.env.ANTHROPIC_MODEL ?? "claude-sonnet-4-6";

export interface LLMClient {
  classifyAndExtract(item: InboxItem): Promise<ClassificationResult>;
  draftReply(input: DraftInput): Promise<string>;
}

export async function classifyAndExtract(
  item: InboxItem,
  llm: LLMClient = createLLMClient(),
): Promise<ClassificationResult> {
  return llm.classifyAndExtract(item);
}

function createLLMClient(): LLMClient {
  const anthropic = new Anthropic();

  return {
    async classifyAndExtract(item: InboxItem): Promise<ClassificationResult> {
      const response = await anthropic.messages.parse({
        model: DEFAULT_MODEL,
        max_tokens: 16000,
        thinking: { type: "adaptive" },
        system: [
          {
            type: "text",
            text: CLASSIFY_SYSTEM_PROMPT,
            cache_control: { type: "ephemeral" },
          },
        ],
        output_config: {
          effort: "high",
          format: zodOutputFormat(ClassificationResultSchema),
        },
        messages: [
          {
            role: "user",
            content: renderItemForClassification(item),
          },
        ],
      });

      if (!response.parsed_output) {
        throw new Error(
          `classifyAndExtract(${item.id}): model did not return parseable structured output (stop_reason=${response.stop_reason})`,
        );
      }
      return response.parsed_output;
    },

    async draftReply(input: DraftInput): Promise<string> {
      const response = await anthropic.messages.create({
        model: DEFAULT_MODEL,
        max_tokens: 300,
        system: [
          {
            type: "text",
            text: DRAFT_SYSTEM_PROMPT,
            cache_control: { type: "ephemeral" },
          },
        ],
        messages: [
          { role: "user", content: renderDraftContext(input) },
        ],
      });
      for (const block of response.content) {
        if (block.type === "text") {
          return block.text.trim();
        }
      }
      throw new Error(
        `draftReply(${input.item.id}): no text block returned (stop_reason=${response.stop_reason})`,
      );
    },
  };
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  const llm = createLLMClient();

  return Promise.all(
    inbox.map((item) =>
      withItemContext(item.id, async () => {
        const classification = await llm.classifyAndExtract(item);
        const routing = await routeItem(item, classification);
        const draft_reply = await draftReply(item, classification, routing, llm);
        return assembleItemOutput(item, classification, routing, draft_reply);
      }),
    ),
  );
}

