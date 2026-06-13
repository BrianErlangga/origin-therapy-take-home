import assert from "node:assert/strict";
import { test } from "node:test";
import {
  groundChildName,
  groundDobOrAge,
  groundMemberId,
  groundParentContact,
  groundPayer,
  ungroundedFields,
  verifyGrounding,
} from "./agent.js";
import type { ExtractedIntake } from "./types.js";

// ---------------------------------------------------------------------------
// member_id
// ---------------------------------------------------------------------------
test("member_id: grounded when present (tolerates dash/space formatting)", () => {
  assert.equal(groundMemberId("XYZ-123-456", "Member ID: XYZ123456"), true);
  assert.equal(groundMemberId("ABC123", "policy abc 123 on file"), true);
});

test("member_id: a single-character-off fabrication is flagged", () => {
  // Source has XYZ124; model extracted XYZ123 — the silent-corruption case.
  assert.equal(groundMemberId("XYZ123", "Member ID: XYZ124"), false);
});

// ---------------------------------------------------------------------------
// parent_contact
// ---------------------------------------------------------------------------
test("parent_contact: phone grounded across formatting differences", () => {
  assert.equal(
    groundParentContact("(555) 123-4567", "call me at 555.123.4567"),
    true,
  );
});

test("parent_contact: fabricated phone is flagged", () => {
  assert.equal(groundParentContact("555-123-9999", "call 555-123-4567"), false);
});

test("parent_contact: email substring match", () => {
  assert.equal(
    groundParentContact("Sofia@Example.com", "from sofia@example.com"),
    true,
  );
  assert.equal(
    groundParentContact("typo@example.com", "from sofia@example.com"),
    false,
  );
});

// ---------------------------------------------------------------------------
// child_name
// ---------------------------------------------------------------------------
test("child_name: grounded when all tokens appear", () => {
  assert.equal(groundChildName("Emma Rodriguez", "Re: Emma Rodriguez"), true);
});

test("child_name: dropped middle name is fine", () => {
  assert.equal(
    groundChildName("Emma Rodriguez", "patient Emma Grace Rodriguez"),
    true,
  );
});

test("child_name: kept middle name still verified", () => {
  assert.equal(
    groundChildName("Emma Grace Rodriguez", "patient Emma Grace Rodriguez"),
    true,
  );
});

test("child_name: fabricated surname is flagged", () => {
  assert.equal(groundChildName("Emma Johnson", "patient Emma Rodriguez"), false);
});

test("child_name: normalized first name (Bob -> Robert) is flagged", () => {
  assert.equal(groundChildName("Robert Smith", "patient Bob Smith"), false);
});

test("child_name: diacritics are normalized", () => {
  assert.equal(groundChildName("Jose Garcia", "patient José García"), true);
});

test("child_name: substring of a longer word does not count", () => {
  // "ann" must not ground against "annapolis"
  assert.equal(groundChildName("Ann Lee", "from annapolis lee street"), false);
});

// ---------------------------------------------------------------------------
// dob_or_age
// ---------------------------------------------------------------------------
test("dob_or_age: date grounded across separator differences", () => {
  assert.equal(groundDobOrAge("03/09/2002", "DOB 3/9/2002"), true);
  assert.equal(groundDobOrAge("2019-03-15", "born 03/15/2019"), true);
});

test("dob_or_age: fabricated date is flagged", () => {
  assert.equal(groundDobOrAge("03/09/2002", "DOB 03/09/2003"), false);
});

test("dob_or_age: age phrase grounded against a different phrasing", () => {
  // "7 years of age" in source, extracted as "7 years old" — must pass.
  assert.equal(groundDobOrAge("7 years old", "she is 7 years of age"), true);
  assert.equal(groundDobOrAge("4 years old", "4-year-old boy"), true);
});

test("dob_or_age: bare age with no unit word grounds (he is 6)", () => {
  assert.equal(groundDobOrAge("6", "my son Leo, he is 6"), true);
});

test("dob_or_age: non-English age unit grounds (Spanish '5 anos')", () => {
  // Must not depend on English year/month words — the number is the anchor.
  assert.equal(groundDobOrAge("5 anos", "mi hija, tiene 5 anos"), true);
});

test("dob_or_age: a digit only inside a longer number does not ground an age", () => {
  // "7" appears only inside the phone run "7123" — not standalone.
  assert.equal(groundDobOrAge("7 years old", "call 555-7123-000"), false);
});

test("dob_or_age: hallucinated age is flagged even if the number appears elsewhere", () => {
  // Child is really 5; model hallucinates "8 anos". The 8 only appears in an
  // unrelated count ("8 sesiones"), not an age context — must be flagged.
  const src = "Hola, mi hija tiene 5 anos. Tenemos 8 sesiones autorizadas.";
  assert.equal(groundDobOrAge("8 anos", src), false);
  assert.equal(groundDobOrAge("5 anos", src), true);
});

test("dob_or_age: age digit inside a member ID does not ground a hallucinated age", () => {
  // "8" is part of the run "7788" in the member ID — never a standalone 8.
  assert.equal(groundDobOrAge("8 years old", "member ID AET-7788, she is 5"), false);
});

test("dob_or_age: word-form date grounds verbatim (no false positive)", () => {
  // The classifier passes word-form dates through verbatim; they must ground.
  const src = "Date of birth: the third of March, two thousand twenty.";
  assert.equal(groundDobOrAge("the third of March, two thousand twenty", src), true);
});

test("dob_or_age: a word-form date NOT in the source is still flagged", () => {
  assert.equal(
    groundDobOrAge("the fourth of July, two thousand twenty", "born the third of March, two thousand twenty"),
    false,
  );
});

test("dob_or_age: day/month transposition is intentionally NOT flagged", () => {
  // Same numeric components, so equally supported by an ambiguous source.
  // Grounding verifies provenance, not date interpretation — resolving
  // 09/03 vs 03/09 is a downstream job. Documented limitation.
  assert.equal(groundDobOrAge("09/03/2002", "DOB 03/09/2002"), true);
});

// ---------------------------------------------------------------------------
// payer
// ---------------------------------------------------------------------------
test("payer: grounds when significant tokens appear (generic suffix ignored)", () => {
  assert.equal(groundPayer("Aetna PPO", "Insurance is Aetna PPO"), true);
  assert.equal(
    groundPayer("Blue Cross Blue Shield PPO", "Insurance: Blue Cross Blue Shield PPO"),
    true,
  );
  assert.equal(groundPayer("Medicaid", "Tenemos Medicaid"), true);
});

test("payer: a hallucinated carrier (A -> B) is flagged", () => {
  // The costly silent case: source says Aetna, model outputs Kaiser.
  assert.equal(groundPayer("Kaiser Permanente", "Insurance is Aetna PPO"), false);
});

test("payer: matching only the generic suffix does not ground", () => {
  // "PPO" alone must not ground a fabricated carrier.
  assert.equal(groundPayer("Cigna PPO", "we have an Aetna PPO plan"), false);
});

// ---------------------------------------------------------------------------
// verifyGrounding orchestration
// ---------------------------------------------------------------------------
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

test("verifyGrounding: null fields are skipped (trivially grounded)", () => {
  const verdicts = verifyGrounding(intake({}), "anything");
  assert.equal(verdicts.length, 0);
  assert.deepEqual(ungroundedFields(verdicts), []);
});

test("verifyGrounding: mixes grounded and ungrounded fields", () => {
  const verdicts = verifyGrounding(
    intake({ child_name: "Emma Rodriguez", member_id: "XYZ123" }),
    "patient Emma Rodriguez, member XYZ124",
  );
  assert.deepEqual(ungroundedFields(verdicts), ["member_id"]);
});
