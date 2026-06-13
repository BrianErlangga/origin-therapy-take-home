import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeEach, test } from "node:test";
import { runAgent, type ClassificationResult, type LLMClient } from "./agent.js";
import { configureTrace } from "./tools.js";
import type { ExtractedIntake, InboxItem } from "./types.js";

// ---------------------------------------------------------------------------
// Integration tests: drive the full runAgent pipeline with a mock LLM so we can
// force a hallucination and assert the grounding gate's end-to-end behavior
// (fail-closed, retry rescue, safeguarding override). No API calls.
// ---------------------------------------------------------------------------

// Each test writes tool calls to a throwaway trace so recordTool is happy and
// getToolCallsForItem is isolated per test.
beforeEach(() => {
  const dir = mkdtempSync(join(tmpdir(), "pipeline-test-"));
  configureTrace({ path: join(dir, "trace.jsonl") });
});

function intake(partial: Partial<ExtractedIntake>): ExtractedIntake {
  return {
    child_name: null,
    dob_or_age: null,
    parent_contact: null,
    discipline: null,
    diagnosis_or_concern: null,
    payer: null,
    member_id: null,
    ...partial,
  };
}

function cls(partial: Partial<ClassificationResult>): ClassificationResult {
  return {
    extracted_intake: partial.extracted_intake ?? intake({}),
    classification: partial.classification ?? "new_referral",
    urgency: partial.urgency ?? "P2",
    safeguarding: partial.safeguarding ?? false,
    missing_info: partial.missing_info ?? [],
  };
}

// Mock LLM scripted per item id: `first` is the initial extraction, optional
// `retry` is what the grounding-hint retry returns.
function mockLLM(
  scripts: Record<string, { first: ClassificationResult; retry?: ClassificationResult }>,
): LLMClient {
  return {
    async classifyAndExtract(item) {
      return scripts[item.id]!.first;
    },
    async classifyAndExtractWithGroundingHint(item) {
      const s = scripts[item.id]!;
      return s.retry ?? s.first;
    },
    async draftReply() {
      return "Thank you — a team member will follow up shortly.";
    },
  };
}

function item(id: string, body: string, partial: Partial<InboxItem> = {}): InboxItem {
  return {
    id,
    channel: "fax_referral",
    received_at: "2026-04-28T08:00:00-07:00",
    sender: partial.sender ?? "Test Pediatrics fax",
    subject: partial.subject ?? "Referral",
    body,
    attachments: [],
    ...partial,
  };
}

const toolNames = (o: { tools_called: { name: string }[] }) =>
  o.tools_called.map((t) => t.name);

test("clean referral: all fields ground → routes fully", async () => {
  const it = item(
    "clean",
    "Child: Emma Lee. DOB 2018-09-04. Parent 555-0101. Insurance: Aetna PPO, member AET-1234.",
  );
  const llm = mockLLM({
    clean: {
      first: cls({
        classification: "new_referral",
        extracted_intake: intake({
          child_name: "Emma Lee",
          dob_or_age: "2018-09-04",
          parent_contact: "555-0101",
          payer: "Aetna PPO",
          member_id: "AET-1234",
          discipline: ["SLP"],
        }),
      }),
    },
  });

  const [out] = await runAgent([it], llm);
  assert.equal(out!.requires_human_review, true);
  assert.ok(toolNames(out!).includes("verify_insurance"), "verify_insurance should fire");
  assert.ok(out!.task_ids.length > 0, "an intake task should be created");
  assert.match(out!.decision_rationale, /Grounding: 5 high-cost field/);
});

test("hallucinated payer (A->B): fails closed, no tools, payer flagged", async () => {
  const it = item("badpayer", "Insurance: Aetna PPO, member AET-1234. Child Emma Lee, DOB 2018-09-04.");
  const bad = cls({
    classification: "new_referral",
    extracted_intake: intake({
      child_name: "Emma Lee",
      dob_or_age: "2018-09-04",
      payer: "Kaiser Permanente", // not in source
      member_id: "AET-1234",
    }),
  });
  const llm = mockLLM({ badpayer: { first: bad, retry: bad } }); // persists on retry

  const [out] = await runAgent([it], llm);
  assert.equal(out!.tools_called.length, 0, "no tools should fire");
  assert.equal(out!.task_ids.length, 0);
  assert.equal(out!.requires_human_review, true);
  assert.match(out!.decision_rationale, /Grounding check failed.*payer/);
  assert.ok(
    out!.missing_info.some((m) => m.toLowerCase().includes("payer")),
    "payer should be listed in missing_info",
  );
});

test("hallucinated member_id: fails closed", async () => {
  const it = item("badid", "Insurance: Aetna PPO, member AET-1234. Emma Lee, DOB 2018-09-04.");
  const bad = cls({
    extracted_intake: intake({
      child_name: "Emma Lee",
      dob_or_age: "2018-09-04",
      payer: "Aetna PPO",
      member_id: "AET-9999", // source says AET-1234
    }),
  });
  const llm = mockLLM({ badid: { first: bad, retry: bad } });

  const [out] = await runAgent([it], llm);
  assert.equal(out!.tools_called.length, 0);
  assert.equal(out!.requires_human_review, true);
  assert.match(out!.decision_rationale, /Grounding check failed.*member_id/);
});

test("retry rescues a reformatted field → proceeds to routing", async () => {
  const it = item("retry", "Insurance: Aetna PPO, member AET-1234. Emma Lee, DOB 2018-09-04, 555-0101.");
  const common = {
    child_name: "Emma Lee",
    dob_or_age: "2018-09-04",
    parent_contact: "555-0101",
    member_id: "AET-1234",
  };
  const llm = mockLLM({
    retry: {
      first: cls({ extracted_intake: intake({ ...common, payer: "Kaiser" }) }), // ungrounded
      retry: cls({ extracted_intake: intake({ ...common, payer: "Aetna PPO" }) }), // grounded
    },
  });

  const [out] = await runAgent([it], llm);
  assert.ok(toolNames(out!).includes("verify_insurance"), "should proceed after retry");
  assert.doesNotMatch(out!.decision_rationale, /Grounding check failed/);
});

test("safeguarding + ungrounded field: still escalates (not suppressed)", async () => {
  const it = item("safe", "My son seems scared of his dad lately. Please call 555-0102.");
  const bad = cls({
    classification: "safeguarding",
    urgency: "P0",
    safeguarding: true,
    extracted_intake: intake({
      child_name: "Fabricated Name", // not in source
      parent_contact: "555-0102",
    }),
  });
  const llm = mockLLM({ safe: { first: bad, retry: bad } });

  const [out] = await runAgent([it], llm);
  assert.ok(toolNames(out!).includes("escalate"), "escalation must still fire");
  assert.equal(out!.escalation?.severity, "P0");
  assert.equal(out!.requires_human_review, true);
  assert.match(out!.decision_rationale, /safeguarding overrides/);
  assert.ok(
    out!.missing_info.some((m) => m.toLowerCase().includes("name")),
    "ungrounded name should still be flagged",
  );
});

test("all high-cost fields null: nothing to ground, routes normally", async () => {
  const it = item("nulls", "Please send the missing referral paperwork for the school-age child.");
  const llm = mockLLM({
    nulls: { first: cls({ classification: "missing_paperwork", missing_info: ["referral form"] }) },
  });

  const [out] = await runAgent([it], llm);
  assert.equal(out!.requires_human_review, true);
  assert.doesNotMatch(out!.decision_rationale, /Grounding check failed/);
  assert.ok(out!.task_ids.length > 0, "missing_paperwork should create a task");
});
